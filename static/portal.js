    let currentUser = null;
    let allAdvisorClassesCache = [];
    let listRequest=0, studentRequest=0, chatRequest=0;
    const workspaceError = text => { const el=document.getElementById("workspaceError"); el.textContent=text; el.classList.toggle("hidden",!text); };
    // Token chỉ giữ trong bộ nhớ của trang; không ghi vào localStorage.
    async function apiFetch(path, options = {}) {
      const session = currentUser;
      const headers = new Headers(options.headers || {});
      if (currentUser) headers.set('Authorization', 'Bearer ' + currentUser.access_token);
      const controller=new AbortController();
      const timer=setTimeout(()=>controller.abort(),35000);
      let res;
      try { res=await fetch(path, {...options, headers, signal:controller.signal}); }
      finally { clearTimeout(timer); }
      if (currentUser !== session) throw new Error('Phiên làm việc đã thay đổi');
      const read = res.json.bind(res);
      res.json = async () => { const value = await read(); if (currentUser !== session) throw new Error('Phiên làm việc đã thay đổi'); return value; };
      if (res.status === 401) { resetSession(); throw new Error('Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.'); }
      return res;
    }
    const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const safeCode = value => /^[A-Z0-9_-]+$/i.test(String(value)) ? String(value) : '';
    // Không đưa chuỗi do người dùng nhập vào HTML hoặc mã JavaScript.
    function appendText(container, className, text) {
      const el = document.createElement('div'); el.className=className; el.textContent=text; container.append(el);
    }


    let loginPending=false;
    async function handleLogin(e) {
      if (e && e.preventDefault) e.preventDefault();
      const u = document.getElementById('inpUsername').value.trim();
      const p = document.getElementById('inpPassword').value;
      const errBox = document.getElementById('loginError');
      if(loginPending) return;
      loginPending=true;
      const submit=document.querySelector('#loginForm button[type=submit]'); submit.disabled=true;
      errBox.classList.add('hidden');

      try {
        const res = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: u, password: p })
        });
        const data = await res.json();
        if (!res.ok) {
          errBox.innerText = data.detail || 'Đăng nhập thất bại';
          errBox.classList.remove('hidden');
          return;
        }
        currentUser = data;
        document.getElementById('loginScreen').classList.add('hidden');
        document.getElementById('mainApp').classList.remove('hidden');
        await setupWorkspaceByRole();
      } catch (err) {
        if(currentUser) { workspaceError('Không tải được không gian học vụ: '+err.message); return; }
        errBox.innerText = 'Lỗi kết nối tới máy chủ: ' + err.message;
        errBox.classList.remove('hidden');
      } finally { loginPending=false; submit.disabled=false; }
    }

    async function logout() {
      try { if (currentUser) await apiFetch('/api/logout', {method:'POST'}); }
      finally { resetSession(); }
    }
    function resetSession() {
      currentUser = null;
      listRequest++; classRequest++; studentRequest++; chatRequest++;
      chatPending=false; const send=document.getElementById("chatSend"); send.disabled=false; send.textContent="Gửi";
      workspaceError("");
      allAdvisorClassesCache = [];
      for (const id of ['chatMessages','stuTermTable','stuCoursesTable','stuWarningsBox','stuProgressCard','stuGpaBars','stuNextTermBanner','advStudentsTable','admFacultiesTable','selAdvisorClass']) { document.getElementById(id)?.replaceChildren(); }
      document.getElementById('scenarioControls')?.remove();
      document.getElementById('inpUsername').value = '';
      document.getElementById('inpPassword').value = '';
      document.getElementById('mainApp').classList.add('hidden');
      document.getElementById('loginScreen').classList.remove('hidden');
    }

    function switchTab(tab) {
      const vChat = document.getElementById('viewChat');
      const vData = document.getElementById('viewData');
      const bChat = document.getElementById('btnTabChat');
      const bData = document.getElementById('btnTabData');

      if (tab === 'chat') {
        vChat.classList.remove('hidden');
        vData.classList.add('hidden');
        bChat.className = "w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2.5 bg-[#282a2c] text-white transition";
        bData.className = "w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-[#9aa0a6] hover:bg-[#282a2c]/60 transition";
        document.getElementById('headerTitle').innerText = "Trợ lý chatbot";
      } else {
        vChat.classList.add('hidden');
        vData.classList.remove('hidden');
        bData.className = "w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2.5 bg-[#282a2c] text-white transition";
        bChat.className = "w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-[#9aa0a6] hover:bg-[#282a2c]/60 transition";
        document.getElementById('headerTitle').innerText = document.getElementById('lblTabData').innerText;
      }
    }

    async function setupWorkspaceByRole() {
      document.getElementById('txtUserId').innerText = currentUser.user_id;
      document.getElementById('badgeRole').innerText = ({student:'SINH VIÊN',advisor:'CỐ VẤN',admin:'QUẢN TRỊ'})[currentUser.role] || 'TÀI KHOẢN';
      document.getElementById('txtRoleLabel').innerText = currentUser.role_label;
      try {
        const view = await apiFetch('/api/data-view');
        if (view.ok) {
          const v = await view.json();
          document.getElementById('txtRoleLabel').innerText += ` | Dữ liệu mô phỏng`;
        }
      } catch (err) { workspaceError('Không tải được thông tin dữ liệu'); }
      if (!currentUser) return;

      document.getElementById('panelStudentData').classList.add('hidden');
      document.getElementById('panelAdvisorData').classList.add('hidden');
      document.getElementById('panelAdminData').classList.add('hidden');

      switchTab('chat');
      const chatInput = document.getElementById('chatInput');

      if (currentUser.role === 'student') {
        document.getElementById('lblTabData').innerText = "Bảng điểm & Kịch bản";
        chatInput.placeholder = "Nhập câu hỏi (VD: dự đoán điểm kì sau, xem điểm của tôi, quy chế 18...)";
        document.getElementById('chatMessages').innerHTML = `
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-emerald-400 flex items-center justify-center text-xs font-bold shrink-0">✦</div>
            <div class="bg-[#1e1f20] border border-[#333537] text-[#e3e3e3] px-5 py-4 rounded-3xl rounded-tl-sm max-w-2xl text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-semibold text-white">Xin chào Sinh viên ${safeCode(currentUser.user_id)}! 👋</div>
              <div class="text-[#c4c7c5]">Bạn có thể nhập trực tiếp câu hỏi vào ô chat bên dưới để tra cứu <b>Kịch bản GPA kỳ sau</b>, <b>Bảng điểm</b> hoặc <b>Quy chế đào tạo</b>.</div>
            </div>
          </div>
        `;
        document.getElementById('panelStudentData').classList.remove('hidden');
        await loadStudentSelfData(currentUser.user_id);

      } else if (currentUser.role === 'advisor') {
        document.getElementById('lblTabData').innerText = "Quản lý Lớp & Khoa";
        chatInput.placeholder = "Nhập câu hỏi quản lý (VD: xem lớp, tóm tắt lớp IT25-18...)";
        document.getElementById('chatMessages').innerHTML = `
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-emerald-400 flex items-center justify-center text-xs font-bold shrink-0">✦</div>
            <div class="bg-[#1e1f20] border border-[#333537] text-[#e3e3e3] px-5 py-4 rounded-3xl rounded-tl-sm max-w-2xl text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-semibold text-emerald-300">Xin chào Cố vấn Học tập (${safeCode(currentUser.user_id)})! 👋</div>
              <div class="text-[#c4c7c5]">Bạn có thể nhập câu hỏi để tra cứu thông tin lớp, sinh viên nguy cơ cao hoặc gõ <b>"xem lớp"</b>.</div>
            </div>
          </div>
        `;
        document.getElementById('panelAdvisorData').classList.remove('hidden');
        await initAdvisorClasses('');

      } else {
        document.getElementById('lblTabData').innerText = "Thống kê Toàn trường & Lớp";
        chatInput.placeholder = "Nhập câu hỏi quản trị toàn trường (VD: xem lớp, tổng sv từng khóa...)";
        document.getElementById('chatMessages').innerHTML = `
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-emerald-400 flex items-center justify-center text-xs font-bold shrink-0">✦</div>
            <div class="bg-[#1e1f20] border border-[#333537] text-[#e3e3e3] px-5 py-4 rounded-3xl rounded-tl-sm max-w-2xl text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-semibold text-amber-300">Xin chào Quản trị viên Toàn trường (${safeCode(currentUser.user_id)})! 👋</div>
              <div class="text-[#c4c7c5]">Bạn có toàn quyền truy xuất báo cáo 19 Khoa, thống kê vĩ mô toàn trường và tra cứu lớp học.</div>
            </div>
          </div>
        `;
        document.getElementById('panelAdminData').classList.remove('hidden');
        document.getElementById('panelAdvisorData').classList.remove('hidden');
        await initAdminFaculties();
        await initAdvisorClasses('');
      }
    }

    function calcClientNextTermGpa(p) {
      return p ? {nextTerm: p.next_term, nextTc: p.next_tc,
        predCur: p.pred_gpa_cur.toFixed(2), predCumCur: p.pred_cum_cur.toFixed(2),
        predImp: p.pred_gpa_imp.toFixed(2), predCumImp: p.pred_cum_imp.toFixed(2)}
        : {nextTerm: "—", nextTc: "—", predCur: "Chưa có", predCumCur: "Chưa có", predImp: "Chưa có", predCumImp: "Chưa có"};
    }

    async function loadStudentSelfData(targetStudentId) {
      const requestId=++studentRequest, session=currentUser;
      const stuId = targetStudentId || currentUser.user_id;
      const res = await apiFetch(`/api/student/me?user_id=${encodeURIComponent(stuId)}`);
      if(requestId!==studentRequest||currentUser!==session) return;
      if (!res.ok) { workspaceError('Không thể truy cập hồ sơ sinh viên.'); return; }
      const data = await res.json();
      if(requestId!==studentRequest||currentUser!==session) return;
      const latest = data.latest;
      const proj = calcClientNextTermGpa(data.gpa_projection);
      const active = latest.state_as_of === 'ENROLLED' && !!data.gpa_projection;
      document.getElementById('stuNextTermBanner').classList.toggle('hidden', !active);
      document.getElementById('stuPredBadge').classList.toggle('hidden', !active);

      document.getElementById('panelStudentData').classList.remove('hidden');
      document.getElementById('stuHeaderInfo').innerText = `🎓 Hồ sơ Sinh viên: ${safeCode(latest.student_id)} • Lớp: ${latest.class_code}`;
      document.getElementById('stuSubInfo').innerText = `Ngành: ${latest.major_name} (${latest.degree_type === 'Bachelor' ? 'Cử nhân' : latest.degree_type === 'Engineer' ? 'Kỹ sư' : latest.degree_type} - Hệ ${latest.education_mode})`;
      document.getElementById('stuPredBadge').innerText = `🔮 Kịch bản GPA HK${proj.nextTerm}: ${proj.predCur} ➔ ${proj.predImp}/4.00`;
      if (latest.graduation_review_date) document.getElementById('stuSubInfo').innerText += ` • Ngày xét tốt nghiệp: ${latest.graduation_review_date}`;
      document.getElementById('stuRiskBadge').innerText = data.risk_display;
      document.getElementById('stuRiskBadge').title = `Mốc hồ sơ: ${latest.snapshot_cutoff}. Mốc dự báo: ${latest.prediction_cutoff || 'Chưa có'}. Dữ liệu mô phỏng đến ${latest.data_as_of}.`;

      document.getElementById('stuNextTermBanner').innerHTML = `
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="font-bold text-blue-300 text-sm">🔮 Kịch bản GPA (HK${proj.nextTerm} - ${proj.nextTc} tín chỉ mới, không học lại)</div>
          <button type="button" data-action="askFromData" data-arg="Kịch bản GPA kỳ sau của sinh viên ${safeCode(latest.student_id)}"
            class="px-3 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition">
            Xem chi tiết Công thức & Kịch bản trên Chat ➔
          </button>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          <div class="bg-[#131314] border border-[#333537] rounded-xl p-3">
            <div class="text-[#9aa0a6] text-[11px]">Kịch bản A: GPA kỳ mới bằng GPA tích lũy</div>
            <div class="text-sm font-bold text-amber-300 mt-0.5">GPA HK${proj.nextTerm} dự kiến: ${proj.predCur}/4.00 • GPA Tích lũy: ${proj.predCumCur}/4.00</div>
          </div>
          <div class="bg-[#131314] border border-emerald-500/30 rounded-xl p-3">
            <div class="text-[#9aa0a6] text-[11px]">Kịch bản B: GPA kỳ mới cao hơn 0,50 (tối đa 4)</div>
            <div class="text-sm font-bold text-emerald-300 mt-0.5">GPA HK${proj.nextTerm} dự kiến: ${proj.predImp}/4.00 • GPA Tích lũy: ${proj.predCumImp}/4.00</div>
          </div>
        </div>
      `;

      document.getElementById('stuTermTable').innerHTML = (data.history || []).map(r => `
        <tr class="hover:bg-[#282a2c]/50">
          <td class="py-2 px-2.5 font-bold text-blue-300">HK${r.snapshot_term_index}</td>
          <td class="py-2 px-2.5 font-bold ${r.gpa4_term < 1.0 ? 'text-rose-400' : (r.gpa4_term < 2.0 ? 'text-amber-300' : 'text-emerald-300')}">${fmtNumber(r.gpa4_term,2)}</td>
          <td class="py-2 px-2.5 font-semibold">${fmtNumber(r.gpa4_cumulative,2)}</td>
          <td class="py-2 px-2.5">${r.credits_earned_cumulative ?? "Chưa có"}/${r.credits_required ?? "Chưa có"} TC</td>
          <td class="py-2 px-2.5 ${r.credits_failed_term > 0 ? 'text-rose-400 font-semibold' : 'text-slate-400'}">${r.credits_failed_term ?? "Chưa có"} TC (${r.failed_credit_ratio == null ? "Chưa có" : Math.round(r.failed_credit_ratio * 100)+"%"})</td>
          <td class="py-2 px-2.5 ${r.credits_owed_cumulative > 0 ? 'text-amber-300 font-bold' : 'text-slate-400'}">${r.credits_owed_cumulative ?? "Chưa có"} TC</td>
          <td class="py-2 px-2.5">${fmtNumber(r.attendance_rate_term,0)}%</td>
          <td class="py-2 px-2.5">${r.tuition_debt_end_vnd == null ? 'Chưa có' : Number(r.tuition_debt_end_vnd).toLocaleString('vi-VN')}đ</td>
        </tr>
      `).join('');

      const historyBars = (data.history || []).map(r => {
        const g = r.gpa4_term;
        const h = g == null ? 0 : Math.max(0, Math.min(100, Math.round((g / 4.0) * 100)));
        const col = g < 1.0 ? 'bg-rose-500' : (g < 2.0 ? 'bg-amber-400' : 'bg-emerald-400');
        return `
          <div class="flex flex-col items-center flex-1">
            <span class="text-xs font-bold mb-1">${fmtNumber(g,2)}</span>
            <div class="w-10 bg-[#131314] h-24 rounded-xl flex items-end p-1 border border-[#333537]">
              <div class="${col} w-full rounded-lg" style="height:${h}%"></div>
            </div>
            <span class="text-[11px] text-[#9aa0a6] mt-1">HK${r.snapshot_term_index}</span>
          </div>
        `;
      }).join('');

      const predH = Math.max(0, Math.min(100, Math.round((Number(proj.predCur) / 4.0) * 100)));
      const predBar = `
        <div class="flex flex-col items-center flex-1">
          <span class="text-xs font-bold text-blue-300 mb-1">${proj.predCur}</span>
          <div class="w-10 bg-[#131314] h-24 rounded-xl flex items-end p-1 border border-dashed border-blue-400">
            <div class="bg-blue-500/80 w-full rounded-lg" style="height:${predH}%"></div>
          </div>
          <span class="text-[11px] text-blue-300 font-semibold mt-1">HK${proj.nextTerm} (Giả định)</span>
        </div>
      `;
      document.getElementById('stuGpaBars').innerHTML = historyBars + (active ? predBar : '');

      const earn = latest.credits_earned_cumulative;
      const req = latest.credits_required;
      const progPct = earn == null || !req ? null : Math.min(100, Math.round((earn / req) * 100));
      const passPct = latest.failed_credit_ratio == null ? null : Math.max(0, Math.round((1 - latest.failed_credit_ratio) * 100));
      const attPct = latest.attendance_rate_term == null ? null : Math.min(100, Math.round(latest.attendance_rate_term));

      document.getElementById('stuProgressCard').innerHTML = `
        <div class="font-bold text-amber-300">Tỷ lệ học tập trọng tâm (Kỳ hiện tại)</div>
        <div>
          <div class="flex justify-between text-[11px] mb-1"><span>Tiến độ tích lũy tín chỉ</span><b>${progPct == null ? "Chưa có dữ liệu" : `${earn}/${req} TC (${progPct}%)`}</b></div>
          <div class="w-full bg-[#131314] h-2.5 rounded-full overflow-hidden"><div class="bg-blue-500 h-full" style="width:${progPct ?? 0}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-[11px] mb-1"><span>Tỷ lệ tín chỉ đạt học kỳ gần nhất</span><b>${passPct == null ? "Chưa có dữ liệu" : `${passPct}%`}</b></div>
          <div class="w-full bg-[#131314] h-2.5 rounded-full overflow-hidden"><div class="bg-emerald-500 h-full" style="width:${passPct ?? 0}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-[11px] mb-1"><span>Tỷ lệ chuyên cần học kỳ (không phải điều kiện thi từng môn)</span><b>${attPct == null ? "Chưa có dữ liệu" : `${attPct}%`}</b></div>
          <div class="w-full bg-[#131314] h-2.5 rounded-full overflow-hidden"><div class="bg-indigo-400 h-full" style="width:${attPct ?? 0}%"></div></div>
        </div>
      `;

      document.getElementById('stuWarningsBox').innerHTML = (data.warnings || []).map(w => `
        <div class="flex items-start gap-2 bg-[#131314] border border-[#333537] rounded-xl p-2.5 text-xs">
          <span>${w.level === 'danger' ? '🔴' : (w.level === 'warning' ? '🟡' : (w.level === 'info' ? '🔵' : '🟢'))}</span>
          <div><b class="text-white">${esc(w.title)}:</b> <span class="text-[#c4c7c5]">${esc(w.detail)}</span></div>
        </div>
      `).join('');

      document.getElementById('stuCoursesTable').innerHTML =
        !(data.courses || []).length
        ? `<tr><td colspan="9" class="p-4 text-amber-300">${
            data.courses_status === 'unavailable'
              ? 'Chưa có dữ liệu bảng điểm học phần.'
              : 'Không tìm thấy học phần phù hợp.'
          }</td></tr>`
        : data.courses.map(c => `
        <tr class="hover:bg-[#282a2c]/50">
          <td class="py-2 px-2 font-bold text-blue-300 whitespace-nowrap">HK${c.term}</td>
          <td class="py-2 px-2 font-mono text-xs text-[#9aa0a6] whitespace-nowrap" title="${safeCode(c.course_id)}">${esc(c.short_code || c.course_id)}${active && c.gpa_eligible && ['F','D','D+','C','C+'].includes(c.grade_letter) ? `<button type="button" class="block text-blue-300" data-action="askFromData" data-arg="Kịch bản học lại ${safeCode(c.course_id)} đạt B của sinh viên ${safeCode(latest.student_id)}">Giả định đạt B</button>` : ''}</td>
          <td class="py-2 px-2 font-medium text-white">${esc(c.course_name || c.course_id)}</td>
          <td class="py-2 px-2 text-center whitespace-nowrap">${esc(c.credits ?? '—')}</td>
          <td class="py-2 px-2 text-center">${esc(c.process_score ?? '—')}</td>
          <td class="py-2 px-2 text-center">${esc(c.exam_score ?? '—')}</td>
          <td class="py-2 px-2 text-center font-bold">${esc(c.final_score ?? '—')}</td>
          <td class="py-2 px-2 text-center font-bold ${c.grade_letter === 'F' ? 'text-rose-400' : 'text-emerald-300'}">${esc(c.grade_letter ?? '—')}</td>
          <td class="py-2 px-2 text-right whitespace-nowrap">${c.eligibility_review ? 'Cần đối chiếu điều kiện thi' : c.exam_eligible === false ? 'Cấm thi' : c.is_passed == null ? 'Chưa có kết quả' : c.is_passed ? '<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 font-semibold">Đạt</span>' : '<span class="px-2 py-0.5 rounded bg-rose-950 text-rose-300 font-bold">Học lại (F)</span>'}${c.eligibility_review || c.grade_policy_review ? ' · Cần đối chiếu quy chế' : ''}</td>
        </tr>
      `).join('');
      showScenarioControls(data, stuId);
    }

    function showScenarioControls(data, stuId) {
      const old = document.getElementById('scenarioControls');
      if (old) old.remove();
      const latest = data.latest;
      if (latest.state_as_of !== 'ENROLLED') return;
      const box = document.createElement('section');
      box.id = 'scenarioControls';
      box.className = 'bg-[#1e1f20] border border-[#333537] rounded-3xl p-5 space-y-3 text-xs';
      const title = document.createElement('h3');
      title.className = 'font-bold text-blue-300 text-sm'; title.textContent = 'Kịch bản GPA tùy chọn'; box.append(title);
      const note = document.createElement('p'); note.textContent = 'Chọn các học phần cần cải thiện và số tín chỉ môn mới. Không cộng trùng tín chỉ; đây không phải dự báo điểm.'; box.append(note);
      const picks = document.createElement('div'); picks.className = 'max-h-48 overflow-y-auto space-y-1'; box.append(picks);
      const grades = ['F','D','D+','C','C+','B','B+','A','A+'];
      const best = new Map();
      for (const c of data.courses || []) {
        if (!c.gpa_eligible || !grades.includes(c.grade_letter)) continue;
        if (!best.has(c.course_id) || grades.indexOf(c.grade_letter) > grades.indexOf(best.get(c.course_id).grade_letter)) best.set(c.course_id,c);
      }
      const selects = new Map();
      for (const c of best.values()) {
        if (['A','A+'].includes(c.grade_letter)) continue;
        const label = document.createElement('label'); label.className = 'flex items-center justify-between gap-3';
        const text = document.createElement('span'); text.textContent = `${c.short_code || c.course_id} (${c.credits} TC, ${c.grade_letter})`;
        const select = document.createElement('select'); select.className = 'bg-[#131314] border border-[#333537] p-1 rounded';
        select.append(new Option('Không chọn',''));
        for (const grade of grades.slice(grades.indexOf(c.grade_letter)+1)) select.append(new Option(grade,grade));
        label.append(text,select); picks.append(label); selects.set(c.course_id,select);
      }
      const controls = document.createElement('div'); controls.className = 'flex flex-wrap gap-3 items-center'; box.append(controls);
      const inputNumber = (label,min,max,step) => {
        const l = document.createElement('label'); l.textContent = label+' ';
        const el = document.createElement('input'); el.type='number'; el.min=min; el.max=max; el.step=step; el.value='0'; el.className='w-20 bg-[#131314] border border-[#333537] rounded p-1'; l.append(el); controls.append(l); return el;
      };
      const credits = inputNumber('Tín chỉ môn mới',0,Math.min(latest.new_course_credits_available ?? 0,latest.registration_max_credits ?? 0,Math.max(0,(latest.credits_required ?? 0)-(latest.credits_earned_cumulative ?? 0)-(latest.credits_owed_cumulative ?? 0)),30),1), gpa = inputNumber('GPA môn mới',0,4,.1);
      const status = document.createElement('p'); status.className='text-emerald-300';
      const post = async (path,payload,render,out) => {
        out.textContent='Đang tính…';
        try {
          const response=await apiFetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:stuId,...payload})});
          const value=await response.json();
          if (!response.ok) throw new Error(typeof value.detail==='string'?value.detail:'Thông tin nhập không hợp lệ');
          out.textContent=render(value);
        } catch (error) { out.textContent=error.message; }
      };
      const btn = document.createElement('button'); btn.type='button';btn.className='bg-blue-600 rounded-lg px-3 py-2 font-bold';btn.textContent='Tính GPA';controls.append(btn);
      btn.onclick=()=>post('/api/student/gpa-scenario',{changes:Object.fromEntries([...selects].filter(([_,v])=>v.value).map(([k,v])=>[k,v.value])),new_credits:Number(credits.value),new_gpa:Number(gpa.value)},v=>`GPA: ${v.gpa_before.toFixed(2)} → ${v.gpa_after.toFixed(2)}; ${v.gpa_credits} tín chỉ tính GPA`,status);
      box.append(status);
      if (data.research_model && latest.prediction_status==='VALID' && latest.tuition_debt_end_vnd>0) {
        const h = document.createElement('h3');h.className='font-bold text-blue-300';h.textContent='Kịch bản ML: thanh toán học phí';box.append(h);
        const group=document.createElement('div');group.className='flex gap-2 items-center';
        const amount=document.createElement('input');amount.type='number';amount.min='1';amount.step='0.01';amount.max=latest.tuition_debt_end_vnd;amount.value=latest.tuition_debt_end_vnd;amount.className='bg-[#131314] border border-[#333537] rounded p-2';amount.setAttribute('aria-label','Số tiền thanh toán giả định (đồng)');
        const b=document.createElement('button');b.type='button';b.textContent='Thử thanh toán (đồng)';b.className='bg-blue-600 rounded-lg px-3 py-2';const o=document.createElement('p');
        b.onclick=()=>post('/api/student/payment-scenario',{payment:Number(amount.value)},v=>`${v.before.risk_percent}% → ${v.after.risk_percent}% (${v.difference_pp} điểm phần trăm). ${v.note}`,o);
        group.append(amount,b);box.append(group,o);
      }
      document.getElementById('panelStudentData').append(box);
    }

    async function inspectStudentInTab(studentId) {
      await loadStudentSelfData(studentId);
      document.getElementById('panelStudentData').scrollIntoView({ behavior: 'smooth' });
    }

    async function initAdvisorClasses(majorCode = '') {
      const req=++listRequest, session=currentUser;
      classRequest++; studentRequest++;
      allAdvisorClassesCache=[];renderClassDropdownOptions([]);updateQuickClasses();
      document.getElementById('advStudentsTable').replaceChildren();
      document.getElementById('advClassSummaryText').textContent='Đang tải danh sách lớp…';
      const url = `/api/advisor/classes` + (majorCode ? `?major_code=${encodeURIComponent(majorCode)}` : '');
      let res;
      try { res=await apiFetch(url); } catch(err) { if(req===listRequest&&session===currentUser) workspaceError('Không tải được danh sách lớp: '+err.message); return; }
      if(req!==listRequest||currentUser!==session) return;
      if (!res.ok) { document.getElementById('advClassSummaryText').textContent='Không thể tải danh sách lớp.'; document.getElementById('advStudentsTable').replaceChildren(); return; }
      const data = await res.json();
      if(req!==listRequest||currentUser!==session) return;
      allAdvisorClassesCache = data.classes || [];updateQuickClasses();
      renderClassDropdownOptions(allAdvisorClassesCache);

      const sel = document.getElementById('selAdvisorClass');
      const hasIT2518 = allAdvisorClassesCache.some(c => c.class_code === 'IT25-18');
      if (!allAdvisorClassesCache.length) { document.getElementById('advClassSummaryText').textContent='Không có lớp được phân công'; return; }
      const defaultCls = (!majorCode && hasIT2518) ? 'IT25-18' : allAdvisorClassesCache[0].class_code;
      const idx=filteredClasses.findIndex(c=>c.class_code===defaultCls);
      classPage=Math.max(0,Math.floor(idx/classPageSize)); renderClassDropdownOptions(filteredClasses,false);
      sel.value=defaultCls;
      await loadAdvisorClassDetail(defaultCls);
      if (majorCode) {
        document.getElementById('panelAdvisorData').scrollIntoView({ behavior: 'smooth' });
      }
    }

    function updateQuickClasses(){ for(const el of document.querySelectorAll('[data-quick-class]')) el.classList.toggle('hidden',!allAdvisorClassesCache.some(c=>c.class_code===el.dataset.quickClass)); }
    let filteredClasses = [], classPage = 0, classRequest = 0;
    const classPageSize = 50;
    function changeClassPage(delta) {
      classPage = Math.max(0, Math.min(Math.max(0,Math.ceil(filteredClasses.length/classPageSize)-1),classPage+delta));
      renderClassDropdownOptions(filteredClasses,false);
      const sel=document.getElementById('selAdvisorClass');
      if (sel.value) loadAdvisorClassDetail(sel.value);
    }
    function renderClassDropdownOptions(list, reset=true) {
      if (reset) { filteredClasses=list; classPage=0; }
      document.getElementById('classPageLabel').textContent=`${list.length} lớp · Trang ${classPage+1}/${Math.max(1,Math.ceil(list.length/classPageSize))}`;
      const sel = document.getElementById('selAdvisorClass');
      sel.innerHTML = list.slice(classPage*classPageSize,(classPage+1)*classPageSize).map(c =>
        `<option value="${esc(c.class_code)}">${esc(c.class_code)} - ${esc(c.major_name)} (${c.n_students} SV | Có dự báo: ${c.valid_predictions ?? "?"} | Cần hỗ trợ: ${c.high_risk_count})</option>`
      ).join('');
    }

    function filterClassDropdown(keyword) {
      const kw = keyword.trim().toUpperCase();
      const filtered = kw ? allAdvisorClassesCache.filter(c => c.class_code.toUpperCase().includes(kw) || c.major_name.toUpperCase().includes(kw)) : allAdvisorClassesCache;
      renderClassDropdownOptions(filtered);
      if (filtered.length > 0) {
        loadAdvisorClassDetail(filtered[0].class_code);
      } else { classRequest++; document.getElementById("advClassSummaryText").textContent="Không có lớp phù hợp"; document.getElementById("advStudentsTable").replaceChildren(); }
    }

    function selectClassQuick(classCode) {
      const idx=allAdvisorClassesCache.findIndex(c=>c.class_code===classCode);
      if(idx<0) return;
      filteredClasses=allAdvisorClassesCache; classPage=Math.floor(idx/classPageSize); renderClassDropdownOptions(filteredClasses,false);
      const sel = document.getElementById('selAdvisorClass');
      sel.value = classCode;
      loadAdvisorClassDetail(classCode);
    }

    async function loadAdvisorClassDetail(classCode) {
      const req=++classRequest; const session=currentUser;
      studentRequest++;
      if(currentUser?.role!=="student") document.getElementById("panelStudentData").classList.add("hidden");
      document.getElementById('advStudentsTable').replaceChildren();
      document.getElementById('advClassSummaryText').textContent='Đang tải lớp…';
      let res;
      try { res=await apiFetch(`/api/advisor/class/${encodeURIComponent(classCode)}`); } catch(err) { if(req===classRequest&&session===currentUser) workspaceError('Không tải được lớp: '+err.message); return; }
      if (req!==classRequest || currentUser!==session) return;
      if (!res.ok) { document.getElementById('advClassSummaryText').textContent='Không có quyền hoặc không tìm thấy lớp.'; document.getElementById('advStudentsTable').replaceChildren(); return; }
      const d = await res.json();
      if (req!==classRequest || currentUser!==session) return;
      document.getElementById('advClassSummaryText').innerText = d.summary_text || '';
      document.getElementById('advStudentsTable').innerHTML = (d.students || []).map(s => {
        const predGpa = s.gpa_projection ? s.gpa_projection.pred_gpa_cur.toFixed(2) : "Chưa có";
        return `
        <tr class="hover:bg-[#282a2c]/50">
          <td class="py-2 px-2.5 font-bold text-blue-300">${safeCode(s.student_id)}</td>
          <td class="py-2 px-2.5">HK${s.snapshot_term_index}</td>
          <td class="py-2 px-2.5">${fmtNumber(s.gpa4_term,2)} / <b>${fmtNumber(s.gpa4_cumulative,2)}</b></td>
          <td class="py-2 px-2.5 font-bold text-emerald-300">${predGpa} / 4.00</td>
          <td class="py-2 px-2.5">${fmtNumber(s.credits_earned_cumulative,0)} TC</td>
          <td class="py-2 px-2.5 ${s.credits_owed_cumulative > 0 ? 'text-amber-300 font-bold' : 'text-slate-400'}">${fmtNumber(s.credits_owed_cumulative,0)} TC</td>
          <td class="py-2 px-2.5">${fmtNumber(s.attendance_rate_term,0)}%</td>
          <td class="py-2 px-2.5 font-bold ${s.prediction_status !== 'VALID' ? 'text-slate-400' : 'text-blue-300'}">${esc(s.risk_display)}</td>
          <td class="py-2 px-2.5 text-right space-x-1 whitespace-nowrap">
            <button type="button" data-action="inspectStudentInTab" data-arg="${safeCode(s.student_id)}"
              class="px-2.5 py-1 rounded-lg bg-emerald-950 hover:bg-emerald-900 text-emerald-300 font-semibold">📊 Bảng điểm</button>
            <button type="button" data-action="askFromData" data-arg="Kịch bản GPA kỳ sau của sinh viên ${safeCode(s.student_id)}"
              class="px-2.5 py-1 rounded-lg bg-blue-950 hover:bg-blue-900 text-blue-300 font-semibold">🔮 Kịch bản GPA</button>
          </td>
        </tr>`;
      }).join('');
    }

    function fmtNumber(value, digits=0) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "Chưa có"; }

    async function initAdminFaculties() {
      const res = await apiFetch(`/api/admin/faculties`);
      if (!res.ok) { document.getElementById('admFacultiesTable').textContent='Không thể tải thống kê khoa.'; return; }
      const data = await res.json();
      document.getElementById('admFacultiesTable').innerHTML = (data.faculties || []).map(f => `
        <tr class="hover:bg-[#282a2c]/50">
          <td class="py-2 px-2.5 font-bold text-amber-300">${safeCode(f.major_code)}</td>
          <td class="py-2 px-2.5 font-medium text-white">${esc(f.major_name)}</td>
          <td class="py-2 px-2.5">${f.n_classes} lớp</td>
          <td class="py-2 px-2.5">${f.n_students.toLocaleString('vi-VN')} SV</td>
          <td class="py-2 px-2.5 font-semibold">${fmtNumber(f.gpa_mean,2)}</td>
          <td class="py-2 px-2.5 text-amber-300 font-semibold">${f.warning_count} SV</td>
          <td class="py-2 px-2.5 text-rose-400 font-bold">${f.valid_predictions ? `${f.high_risk_count}/${f.valid_predictions} (${f.high_risk_pct}%)` : "Chưa có dự báo"}<br><small>Độ bao phủ: ${f.valid_predictions ?? 0}/${f.n_students}</small></td>
          <td class="py-2 px-2.5">${fmtNumber(f.total_debt_billion,2)} tỷ</td>
          <td class="py-2 px-2.5 text-right space-x-1 whitespace-nowrap">
            <button type="button" data-action="initAdvisorClasses" data-arg="${safeCode(f.major_code)}"
              class="px-2.5 py-1 rounded-lg bg-emerald-950 hover:bg-emerald-900 text-emerald-300 font-semibold">🏫 Xem Lớp & SV</button>
            <button type="button" data-action="askFromData" data-arg="Tổng kết Khoa ${safeCode(f.major_code)}"
              class="px-2.5 py-1 rounded-lg bg-amber-950 hover:bg-amber-900 text-amber-300 font-semibold">📊 Báo cáo Khoa</button>
          </td>
        </tr>
      `).join('');
    }

    function askFromData(questionText) {
      switchTab('chat');
      document.getElementById('chatInput').value = questionText;
      sendChatMessage(new Event('submit'));
    }

    let chatPending = false;
    async function sendChatMessage(e) {
      if (e && e.preventDefault) e.preventDefault();
      const inp = document.getElementById('chatInput');
      const msg = inp.value.trim();
      if (!msg || !currentUser || chatPending) return;
      chatPending = true;
      const req=++chatRequest;
      const send = document.getElementById('chatSend');
      send.disabled = true; send.textContent = 'Đang xử lý…';
      inp.value = '';

      const box = document.getElementById('chatMessages');
      const session = currentUser;
      const line = document.createElement('div'); line.className='flex justify-end';
      appendText(line,'bg-blue-600 text-white px-4 py-2.5 rounded-3xl rounded-tr-sm max-w-xl text-xs',msg);
      box.append(line);
      box.scrollTop = box.scrollHeight;

      try {
        const res = await apiFetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: msg
          })
        });
        const data = await res.json();
        if (currentUser !== session) return;
        if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Không thể xử lý yêu cầu");
        box.insertAdjacentHTML("beforeend", `
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-emerald-400 flex items-center justify-center text-xs font-bold shrink-0">✦</div>
            <div class="bg-[#1e1f20] border border-[#333537] text-[#e3e3e3] px-5 py-4 rounded-3xl rounded-tl-sm max-w-3xl w-full text-xs leading-relaxed">
              ${data.reply || 'Không có phản hồi.'}
            </div>
          </div>
        `);
        box.scrollTop = box.scrollHeight;
      } catch (err) {
        if (currentUser === session) appendText(box,'text-xs text-rose-400',err.message);
      } finally {
        if (req===chatRequest) { chatPending = false; send.disabled = false; send.textContent = 'Gửi'; }
      }
    }

    // Danh sách handler cố định; tuyệt đối không eval nội dung thuộc tính.
    const actions={switchTab, initAdvisorClasses, changeClassPage: v=>changeClassPage(Number(v)), selectClassQuick, askFromData, inspectStudentInTab, logout};
    function handleAction(fn,arg) {
      const session=currentUser;
      Promise.resolve().then(()=>fn(arg)).catch(err=>{if(session===currentUser) workspaceError(err.message);});
    }
    document.addEventListener('click',event=>{
      const el=event.target.closest('[data-action]');
      if(el && Object.hasOwn(actions,el.dataset.action)) handleAction(actions[el.dataset.action],el.dataset.arg);
    });
    document.addEventListener('submit',event=>{
      const key=event.target.dataset.submit;
      if(key==='login') {event.preventDefault();handleLogin(event);}
      if(key==='chat') {event.preventDefault();sendChatMessage(event);}
    });
    document.addEventListener('input',event=>{if(event.target.dataset.input==='filter') filterClassDropdown(event.target.value);});
    document.addEventListener('change',event=>{if(event.target.dataset.change==='class') handleAction(loadAdvisorClassDetail,event.target.value);});
