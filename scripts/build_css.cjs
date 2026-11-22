// Chỉ cần Node khi thay đổi giao diện; file CSS đã biên dịch được lưu trong Git.
const fs = require('fs');
const path = require('path');
const twPath = path.dirname(require.resolve('tailwindcss/package.json'));
const {compile} = require('tailwindcss');
(async () => {
  const compiler = await compile(fs.readFileSync(path.join(twPath,'index.css'),'utf8'), {
    base:twPath, loadStylesheet:async (id,base) => ({content:fs.readFileSync(path.resolve(base,id),'utf8'),base:path.dirname(path.resolve(base,id))})
  });
  const files=['templates/index.html',...fs.readdirSync('core').filter(f=>f.endsWith('.py')).map(f=>'core/'+f)];
  const candidates=[...new Set(files.flatMap(f=>fs.readFileSync(f,'utf8').match(/[A-Za-z0-9_!@#%/:.\[\]\-]+/g)||[]))];
  fs.writeFileSync('static/portal.css',compiler.build(candidates).trimEnd()+'\n');
})().catch(e=>{console.error(e.message);process.exitCode=1;});
