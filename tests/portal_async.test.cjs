// Kiểm thử JavaScript thực bằng VM và fetch giả lập (không phải kiểm thử trình duyệt).
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const script=fs.readFileSync(path.join(__dirname,'../static/portal.js'),'utf8');
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
function harness(){
  const nodes=new Map(),timers=new Map();let id=0;
  const element=key=>{if(!nodes.has(key))nodes.set(key,{value:'',textContent:'',innerText:'',classList:{add(){},remove(){},toggle(){}},replaceChildren(){},remove(){},append(){},scrollIntoView(){},insertAdjacentHTML(){}});return nodes.get(key);};
  const ctx=vm.createContext({Headers,AbortController,document:{getElementById:element,querySelector:element,addEventListener(){},createElement:()=>element('new')},setTimeout:fn=>{timers.set(++id,fn);return id},clearTimeout:i=>timers.delete(i)});
  vm.runInContext(script,ctx);
  return {ctx,element,timers,run:code=>vm.runInContext(code,ctx)};
}
const response=(value,status=200)=>({ok:status===200,status,json:async()=>value});
const tick=()=>new Promise(r=>setImmediate(r));

test('Deadline đọc hết JSON, không dừng ở response headers',async()=>{
  const h=harness();let signal;
  h.ctx.fetch=async(_,o)=>{signal=o.signal;return {ok:true,status:200,json:()=>new Promise((_,reject)=>signal.addEventListener('abort',()=>reject(new Error('aborted'))))};};
  const request=h.run('apiFetch("/api/data-view")');await tick();assert.equal(h.timers.size,1);
  const rejected=assert.rejects(request,/quá thời hạn/);[...h.timers.values()][0]();await rejected;
  assert.equal(signal.aborted,true);assert.equal(h.timers.size,0);
});

test('Đăng nhập dùng deadline, không gửi token cũ và không reset sai khi 401',async()=>{
  const h=harness();h.element('inpUsername').value='BIT250093';h.element('inpPassword').value='BIT250093';
  let called=false;h.ctx.fetch=async(p,o)=>{called=true;assert.equal(p,'/api/login');assert.equal(o.headers.has('Authorization'),false);return response({detail:'Sai mật khẩu'},401)};
  await h.run('handleLogin()');assert.equal(called,true);assert.equal(h.element('loginError').innerText,'Sai mật khẩu');assert.equal(h.run('sessionEpoch'),0);assert.equal(h.timers.size,0);
});

test('Phản hồi 401 của phiên cũ không reset phiên mới',async()=>{
  const h=harness();h.run('currentUser={access_token:"a"}');const d=deferred();h.ctx.fetch=()=>d.promise;
  const pending=h.run('apiFetch("/api/data-view")');const rejected=assert.rejects(pending,/Phiên làm việc/);
  h.run('resetSession();currentUser={access_token:"b"};sessionEpoch++');d.resolve(response({},401));await rejected;
  assert.equal(h.run('currentUser.access_token'),'b');assert.equal(h.timers.size,0);
});

test('Logout cũ không thu hồi phiên mới',async()=>{
  const h=harness();h.run('currentUser={access_token:"a"}');const d=deferred();h.ctx.fetch=()=>d.promise;
  const pending=h.run('logout()');const rejected=assert.rejects(pending,/Phiên làm việc/);
  h.run('resetSession();currentUser={access_token:"b"};sessionEpoch++');d.resolve(response({}));await rejected;
  assert.equal(h.run('currentUser.access_token'),'b');
});

test('Đổi lớp: phản hồi cũ không ghi đè lớp mới',async()=>{
  const h=harness();h.run('currentUser={access_token:"a",role:"advisor"}');const a=deferred(),b=deferred();h.ctx.fetch=p=>p.endsWith('IT25-01')?a.promise:b.promise;
  const old=h.run('loadAdvisorClassDetail("IT25-01")'),latest=h.run('loadAdvisorClassDetail("IT25-02")');
  b.resolve(response({summary_text:'Lớp mới',students:[]}));await latest;
  a.resolve(response({summary_text:'Lớp cũ',students:[]}));await old;
  assert.equal(h.element('advClassSummaryText').innerText,'Lớp mới');assert.equal(h.timers.size,0);
});

test('Đọc body dở khi reset phiên bị hủy, không trả JSON cũ',async()=>{
  const h=harness();h.run('currentUser={access_token:"a"}');const d=deferred();let signal;
  h.ctx.fetch=async(_,o)=>{signal=o.signal;return {ok:true,status:200,json:()=>d.promise}};
  const request=h.run('apiFetch("/api/student/me")');await tick();const rejected=assert.rejects(request,/Phiên làm việc/);
  h.run('resetSession();currentUser={access_token:"b"};sessionEpoch++');assert.equal(signal.aborted,true);d.resolve({private:'cũ'});await rejected;
  assert.equal(h.run('currentUser.access_token'),'b');
});

test('Đăng nhập bị treo khi đọc body vẫn hết hạn và mở lại nút gửi',async()=>{
  const h=harness();h.element('inpUsername').value='BIT250093';h.element('inpPassword').value='BIT250093';
  h.ctx.fetch=async(_,o)=>({ok:true,status:200,json:()=>new Promise((_,reject)=>o.signal.addEventListener('abort',()=>reject(new Error('aborted'))))});
  const pending=h.run('handleLogin()');await tick();assert.equal(h.timers.size,1);[...h.timers.values()][0]();await pending;
  assert.match(h.element('loginError').innerText,/quá thời hạn/);assert.equal(h.run('loginPending'),false);assert.equal(h.run('currentUser'),null);
});
