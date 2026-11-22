"""Tra cứu quy chế theo đoạn đã đối chiếu ảnh PDF; BM25 và embedding tùy chọn."""
from collections import Counter
import csv
import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import unicodedata

import numpy as np
from core.chatbot_engine import strip_accents

ROOT = Path(__file__).resolve().parents[1]
PDF = 'Quy-che-dao-tao-trinh-do-dai-hoc.pdf'
PDF_SHA = '10692552bf4245f4ee5a5487cbad69d123ea7bdb2fde6c13d1658d97ed857ea7'
STOP = set('la cua va trong cho ve co duoc hay nhu the nao nhieu gi toi em mot nhung cac khong sinh vien truong cmc theo quy dinh quy che'.split())


def tokens(text):
    return [w for w in re.findall(r'[a-z0-9]+', strip_accents(text)) if w not in STOP]


def numbers(text):
    q = strip_accents(text)
    arts = [int(x) for x in re.findall(r'\b(?:dieu|quy\s*che|quy\s*dinh)\s*(\d{1,2})\b', q)]
    clauses = [int(x) for x in re.findall(r'\bkhoan\s*(\d{1,2})\b', q)]
    return arts, clauses


def span(s):
    ends = [int(x) for x in s.split('–')]
    return range(ends[0], ends[-1]+1)


def safe_api_error(exc):
    """Chỉ ghi mã lỗi, không ghi thông báo từ SDK có thể chứa bí mật."""
    code = getattr(exc, 'code', None)
    if code is None:
        code = getattr(exc, 'status_code', None)
    if isinstance(code, int) and 400 <= code <= 599:
        return f'HTTP_{code}'
    if isinstance(exc, TimeoutError) or 'timeout' in type(exc).__name__.lower():
        return 'TIMEOUT'
    return 'SDK_ERROR'


def finish_status(response):
    candidates = getattr(response, 'candidates', None) or []
    if not candidates:
        return 'NO_CANDIDATES'
    reason = getattr(candidates[0], 'finish_reason', None)
    reason = getattr(reason, 'value', reason)
    # Không ghi dữ liệu tùy ý từ phản hồi vào log.
    return reason if isinstance(reason, str) and reason in {
        'STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION', 'OTHER', 'BLOCKLIST',
        'PROHIBITED_CONTENT', 'SPII', 'MALFORMED_FUNCTION_CALL'
    } else 'UNKNOWN'


class RegulationSearch:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        pdf = self.root/'regulations'/PDF
        with pdf.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != PDF_SHA:
                raise ValueError('PDF quy chế đã thay đổi; phải đối chiếu lại kho bằng chứng.')
        path = self.root/'regulations/quy_che_chuan_hoa.tsv'
        self.corpus_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8',newline='') as f:
            self.chunks = list(csv.DictReader(f,delimiter='\t'))
        for c in self.chunks:
            c.update(article=int(c['article']), id=f"D{c['article']}.K{c['clause']}")
        if {page for c in self.chunks for page in span(c['pages'])} != set(range(5,28)):
            raise ValueError('Kho bằng chứng chưa phủ đủ trang nội dung PDF 5–27')
        if set(c['article'] for c in self.chunks) != set(range(1,28)):
            raise ValueError('Kho quy chế chưa đủ 27 điều')
        if len(set(c['id'] for c in self.chunks)) != len(self.chunks):
            raise ValueError('Trùng đoạn quy chế')
        self.bags = [Counter(tokens(c['title']+' '+c['title']+' '+c['text'])) for c in self.chunks]
        self.lengths = np.array([sum(b.values()) for b in self.bags])
        self.avg = float(self.lengths.mean())
        self.df = Counter(w for b in self.bags for w in b)
        self.client = None
        self.matrix = None
        self.lock = threading.Lock()

    def _cloud(self):
        # Chỉ kích hoạt theo lựa chọn rõ ràng; không gửi hồ sơ, lịch sử chat hoặc token.
        if os.getenv('CMC_RAG_CLOUD') != '1' or not os.getenv('GEMINI_API_KEY'):
            return None
        with self.lock:
            if self.client is None:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=types.HttpOptions(timeout=60000))
        return self.client

    def _embed(self, texts, task):
        from google.genai import types
        response = self._cloud().models.embed_content(model=os.getenv('CMC_EMBED_MODEL','gemini-embedding-001'), contents=texts,
                config=types.EmbedContentConfig(task_type=task,output_dimensionality=768))
        m = np.asarray([x.values for x in response.embeddings],dtype=float)
        if m.shape != (len(texts),768) or not np.isfinite(m).all():
            raise ValueError('Embedding không hợp lệ')
        norms = np.linalg.norm(m,axis=1,keepdims=True)
        if (norms==0).any(): raise ValueError('Embedding rỗng')
        return m/norms

    def _semantic(self, focus):
        with self.lock:
            matrix = self.matrix
        if matrix is None:
            matrix = self._embed([c['title']+' '+c['text'] for c in self.chunks], 'RETRIEVAL_DOCUMENT')
            with self.lock: self.matrix = matrix
        return matrix @ self._embed([focus], 'RETRIEVAL_QUERY')[0]

    def retrieve(self, question, limit=5, cloud=False, trace=None):
        arts, clauses = numbers(question)
        if clauses and not arts: return []
        if len(arts)>1 and clauses: return []
        if arts:
            return [c for c in self.chunks if c['article'] in arts and (not clauses or any(k in span(c['clause']) for k in clauses))][:12]
        query = strip_accents(question)
        query = query.replace('phuc khao','phuc tra').replace('song nganh','hai chuong trinh').replace('thoi gian toi da','thoi gian hoc toi da')
        q = set(tokens(query))
        original = set(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', question.lower())))
        accented = {w for w in original if w != strip_accents(w) and strip_accents(w) in q}
        phrase = ' '.join(tokens(query))
        if not q: return []
        scores = []
        for i,b in enumerate(self.bags):
            score = sum(math.log(1+(len(self.bags)-self.df[w]+.5)/(self.df[w]+.5))*b[w]*2.5/(b[w]+1.5*(.25+.75*self.lengths[i]/self.avg)) for w in q if b[w])
            title = ' '.join(tokens(self.chunks[i]['title']))
            if phrase and len(phrase.split()) >= 2 and phrase in title: score += 12
            if accented:
                original_doc = set(re.findall(r'[^\W_]+',self.chunks[i]['text'].lower()+' '+self.chunks[i]['title'].lower()))
                if not accented & original_doc and 'phuc khao' not in strip_accents(question): score=0
            scores.append(score)
        # Lọc từ hiếm/trùng chủ đề trước, không lấy kết quả ngẫu nhiên khi không có căn cứ.
        ranking = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
        candidates = [i for i in ranking if scores[i]>=max(2.,scores[ranking[0]]*.50) and (len(q & self.bags[i].keys())>=2 or (len(q)==1 and any(len(w)>=4 and self.df[w]<=2 for w in q & self.bags[i].keys())))]
        if not candidates: return []
        if cloud:
            try:
                if self._cloud():
                    # Chỉ gửi tên chủ đề công khai, không gửi nguyên câu hỏi người dùng.
                    focus = '; '.join(self.chunks[i]['title'] for i in candidates[:3])
                    similarities = self._semantic(focus)
                    if trace is not None: trace['embedding_used']=True
                    candidates.sort(key=lambda i:scores[i]/max(scores)+.35*float(similarities[i]),reverse=True)
            except Exception as exc:
                code = safe_api_error(exc)
                if trace is not None: trace['embedding_error'] = code
                logging.getLogger(__name__).warning('Embedding không khả dụng (%s); dùng BM25.', code)
        return [self.chunks[i] for i in candidates[:limit]]

    def answer(self, question, cloud=False):
        trace={'embedding_used':False,'generation_status':'NOT_REQUESTED'}
        chunks = self.retrieve(question,cloud=cloud,trace=trace)
        if not chunks:
            return {'reply':'<p>Chưa tìm thấy căn cứ trong quy chế. Hãy nêu Điều, Khoản hoặc chủ đề cụ thể.</p>','regulations':[], 'method':'none',**trace}
        evidence = [{'id':c['id'],'article':c['article'],'clause':c['clause'],'pages':c['pages'],'document':PDF,'title':c['title'],'text':c['text']} for c in chunks]
        method = 'bm25'
        intro = ''
        if cloud:
            code = 'SDK_ERROR'
            try:
                client = self._cloud()
                trace['generation_status'] = 'DISABLED'
                if client:
                    trace['generation_status'] = 'REQUESTED'
                    from google.genai import types
                    # Không gửi câu hỏi gốc hoặc dữ liệu cá nhân ra ngoài. LLM chỉ diễn giải chủ đề công khai được truy xuất.
                    prompt = ('Tóm lược bằng tiếng Việt các bằng chứng quy chế dưới đây. Không thêm quy định, số liệu hoặc kết luận cá nhân. '
                              'Nếu thiếu căn cứ, nói rõ. Trả JSON gồm answer (chuỗi) và citations (mảng ID chỉ từ bằng chứng).\n'
                              'Chủ đề tra cứu: '+ '; '.join(dict.fromkeys(c['title'] for c in evidence))+'\n'+json.dumps(evidence,ensure_ascii=False))
                    allowed = {c['id'] for c in evidence}
                    schema = {'type':'OBJECT','properties':{
                        'answer':{'type':'STRING'},
                        'citations':{'type':'ARRAY','items':{'type':'STRING','enum':sorted(allowed)}}
                    },'required':['answer','citations']}
                    r = client.models.generate_content(model=os.getenv('CMC_LLM_MODEL','gemini-3.8-flash'),contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type='application/json',response_schema=schema,temperature=0,max_output_tokens=4096))
                    # Một số model dùng chung ngân sách token cho suy luận và đầu ra.
                    status = finish_status(r)
                    if status not in ('STOP','UNKNOWN','NO_CANDIDATES'):
                        code = status
                        raise ValueError('Phản hồi không hoàn tất')
                    code = 'EMPTY_RESPONSE' if status in ('UNKNOWN','STOP','NO_CANDIDATES') else status
                    raw = r.text
                    if not isinstance(raw,str) or not raw.strip():
                        raise ValueError('Phản hồi rỗng')
                    code = 'INVALID_JSON'
                    obj = json.loads(raw)
                    code = 'INVALID_CITATIONS'
                    ids = obj.get('citations') if isinstance(obj,dict) else None
                    if not isinstance(ids,list) or not ids or not all(isinstance(x,str) and x in allowed for x in ids) or not isinstance(obj.get('answer'),str) or not obj['answer'].strip():
                        raise ValueError('Trích dẫn không hợp lệ')
                    intro = '<p><b>Diễn giải AI (đối chiếu bằng chứng bên dưới):</b> '+html.escape(obj['answer'][:6000])+' ['+html.escape(', '.join(ids))+']</p>'
                    method='gemini'
                    trace['generation_status']='OK'
            except Exception as exc:
                # Không ghi exception gốc, nội dung phản hồi, prompt hoặc khóa API.
                if code == 'SDK_ERROR': code = safe_api_error(exc)
                trace['generation_status']=code
                logging.getLogger(__name__).warning('LLM không khả dụng (%s); trả bằng chứng trực tiếp.',code)
        blocks = []
        for c in evidence:
            page = c['pages'].split('–')[0]
            blocks.append(f"<div class='border border-slate-600 rounded-lg p-3'><b>Điều {c['article']}, Khoản {c['clause']}: {html.escape(c['title'])}</b><p>{html.escape(c['text'])}</p><p><a href='/regulations/{PDF}#page={page}' target='_blank' rel='noopener noreferrer'>QĐ 22/2023 · trang PDF {c['pages']}</a></p></div>")
        return {'reply':intro+''.join(blocks)+'<p>Bản tóm lược đã đối chiếu ảnh trang; văn bản PDF gốc là căn cứ. Áp dụng từ năm học 2023–2024.</p>', 'regulations':evidence, 'method':method,**trace}
