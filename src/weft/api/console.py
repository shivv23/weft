"""Self-contained web console for Weft.

A single HTML document (no build step, no CDN dependencies) served from the
FastAPI app at ``/`` and ``/console``. It talks to the REST API using the same
origin, deriving the public prefix from ``location.pathname`` so it works both
behind a reverse proxy that strips ``/weft`` and when run locally.

Deliberately embedded as a string rather than a static file so it ships with the
installed package without needing package-data configuration.
"""

from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weft Console</title>
<style>
:root{--bg:#0b0f14;--panel:#121821;--border:#1e2833;--text:#e6edf3;--muted:#8b98a5;--accent:#5b9dff;--good:#3fb950;--bad:#f85149}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text)}
a{color:var(--accent);text-decoration:none}
header{display:flex;align-items:center;gap:16px;padding:14px 20px;border-bottom:1px solid var(--border);background:var(--panel);position:sticky;top:0;z-index:5;flex-wrap:wrap}
h1{font-size:16px;margin:0}h1 span{color:var(--accent)}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.muted{color:var(--muted)}
input,textarea,select,button{font:inherit;color:var(--text);background:#0e141c;border:1px solid var(--border);border-radius:8px;padding:8px 10px}
input:focus,textarea:focus,select:focus{outline:none;border-color:var(--accent)}
textarea{width:100%;min-height:90px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
button{cursor:pointer;background:#17212d}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:#04101f;border-color:var(--accent);font-weight:600}
nav{display:flex;gap:4px;padding:10px 20px 0;border-bottom:1px solid var(--border);background:var(--panel);flex-wrap:wrap}
nav button{border:1px solid transparent;border-bottom:none;background:transparent;border-radius:8px 8px 0 0;color:var(--muted)}
nav button.active{background:var(--bg);border-color:var(--border);color:var(--text)}
main{padding:20px;max-width:1200px;margin:0 auto}
.panel{display:none}.panel.active{display:block}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:12px;color:var(--muted)}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;border:1px solid var(--border);margin-right:5px}
.badge.episode{color:#7ee787;border-color:#1f6f3d}.badge.fact{color:#79c0ff;border-color:#1f4a7a}
.badge.entity{color:#d2a8ff;border-color:#5a3f8a}.badge.procedure{color:#ffa657;border-color:#7a4a1f}
.chip{display:inline-block;padding:1px 6px;border-radius:6px;font-size:11px;background:#17212d;border:1px solid var(--border);margin:0 4px 4px 0;color:var(--muted)}
.item{border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:10px;background:#0e141c}
.item .text{margin-top:6px;white-space:pre-wrap}
.score{float:right;font-weight:700;color:var(--accent)}
.kv{display:grid;grid-template-columns:170px 1fr;gap:4px 10px;font-size:13px}
.kv b{color:var(--muted);font-weight:500}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--muted);font-weight:500}
pre{background:#0e141c;border:1px solid var(--border);border-radius:8px;padding:10px;overflow:auto;font-size:12.5px}
.msg{border-left:3px solid var(--accent);padding:8px 12px;margin-bottom:8px;background:#0e141c;border-radius:0 8px 8px 0}
.msg.assistant{border-left-color:var(--good)}.msg .meta{color:var(--muted);font-size:12px}
.bar{height:7px;background:#0b0f14;border:1px solid var(--border);border-radius:4px;overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;background:var(--accent)}
#toast{position:fixed;right:16px;bottom:16px;max-width:400px;z-index:99}
.toast{background:#17212d;border:1px solid var(--border);border-left:3px solid var(--accent);padding:10px 14px;border-radius:8px;margin-top:8px;word-break:break-word}
.toast.err{border-left-color:var(--bad)}
.hint{font-size:12.5px;color:var(--muted)}
.hstat{display:inline-block;min-width:64px;text-align:center;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--border)}
</style>
</head>
<body>
<header>
  <h1><span>Weft</span> Console</h1>
  <div class="field"><label>Tenant (X-Tenant-Id)</label><select id="tenant" style="min-width:160px"></select></div>
  <div class="field"><label>API base</label><input id="apibase" readonly style="width:240px"></div>
  <div style="flex:1"></div>
  <a href="docs" target="_blank">Swagger &#8599;</a>
  <a href="openapi.json" target="_blank">openapi.json</a>
</header>
<nav id="tabs">
  <button data-t="recall" class="active">Recall</button>
  <button data-t="sessions">Sessions</button>
  <button data-t="handoffs">Handoffs</button>
  <button data-t="blackboard">Blackboard</button>
  <button data-t="stats">Stats</button>
</nav>
<main>

<section class="panel active" id="p-recall">
  <div class="card">
    <div class="row">
      <div class="field" style="flex:1;min-width:260px"><label>Query</label><input id="rq" placeholder="semantic query"></div>
      <div class="field"><label>k</label><input id="rk" value="5" style="width:70px"></div>
      <div class="field"><label>user_id (optional)</label><input id="ruser" style="width:170px"></div>
      <div class="field"><label>agent_id (optional)</label><select id="ragent" style="min-width:160px"></select></div>
      <button class="primary" onclick="doRecall()">Recall</button>
    </div>
    <p class="hint">Hybrid recall: HNSW vector search + full-text + Cypher traversal, re-ranked by recency, importance and usage, with a score breakdown and provenance path per hit.</p>
  </div>
  <div id="rresults"></div>
</section>

<section class="panel" id="p-sessions">
  <div class="card">
    <div class="row">
      <div class="field"><label>user_id</label><input id="su"></div>
      <div class="field"><label>agent_id (optional)</label><select id="sa"></select></div>
      <div class="field"><label>title (optional)</label><input id="st"></div>
      <button class="primary" onclick="createSession()">Create session</button>
    </div>
    <div class="row" style="margin-top:12px">
      <div class="field" style="flex:1;min-width:240px"><label>Session id</label><input id="sid"></div>
      <div class="field"><label>Load existing</label><select id="slist" onchange="if(this.value){$('sid').value=this.value;loadSession();loadMessages()}"></select></div>
      <button onclick="loadSession()">Load</button>
      <button onclick="loadMessages()">Messages</button>
      <button onclick="loadTimeline()">Timeline</button>
      <button onclick="consolidate()">Consolidate</button>
    </div>
  </div>
  <div class="card" id="sinfo"><span class="hint">No session loaded.</span></div>
  <div class="card">
    <label class="hint">Append messages (JSON array of {role, content, meta?})</label>
    <textarea id="smsg" placeholder='[{"role":"user","content":"..."}]'></textarea>
    <div class="row" style="margin-top:8px"><button class="primary" onclick="appendMsgs()">Append</button></div>
  </div>
  <div id="sout"></div>
</section>

<section class="panel" id="p-handoffs">
  <div class="card">
    <div class="row">
      <div class="field"><label>Your agent id</label><select id="hagent" style="min-width:200px"></select></div>
      <div class="field"><label>inbox status</label><select id="hstatus"><option>pending</option><option>claimed</option><option>completed</option><option>rejected</option></select></div>
      <button class="primary" onclick="loadInbox()">Load inbox</button>
      <button onclick="loadAgents()">List agents</button>
    </div>
    <div class="row" style="margin-top:12px">
      <div class="field"><label>from_agent</label><select id="hfrom"></select></div>
      <div class="field"><label>to_agent</label><select id="hto"></select></div>
      <div class="field" style="flex:1"><label>title</label><input id="htitle"></div>
      <button onclick="createHandoff()">Create handoff</button>
    </div>
    <p class="hint">Claims are atomic: only the first agent to claim succeeds, every other claim gets a 409.</p>
  </div>
  <div id="hout"></div>
</section>

<section class="panel" id="p-blackboard">
  <div class="card">
    <div class="row">
      <div class="field" style="flex:1;min-width:220px"><label>Scopes (exact, comma separated; blank = all)</label><input id="bscopes"></div>
      <button class="primary" onclick="loadScope()">List state</button>
    </div>
    <div class="row" style="margin-top:12px">
      <div class="field"><label>scope</label><input id="bscope" placeholder="global"></div>
      <div class="field"><label>key</label><input id="bkey"></div>
      <div class="field"><label>expected_version (blank = create)</label><input id="bver" style="width:180px"></div>
      <div class="field" style="flex:1;min-width:220px"><label>value (JSON)</label><input id="bval"></div>
      <button onclick="setState()">Set state</button>
    </div>
    <p class="hint">Scopes are matched exactly: use <code>global</code>, <code>session:&lt;id&gt;</code>, <code>user:&lt;id&gt;</code>, <code>agent:&lt;id&gt;</code> or <code>team:&lt;name&gt;</code>. Writes are compare-and-swap: set expected_version to the version you read to avoid lost updates.</p>
  </div>
  <div id="bout"></div>
</section>

<section class="panel" id="p-stats">
  <div class="card">
    <div class="row">
      <button class="primary" onclick="loadStats()">Refresh stats</button>
      <button onclick="loadTenants()">List tenants</button>
    </div>
  </div>
  <div id="stout"></div>
</section>

</main>
<div id="toast"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const base=location.pathname.replace(/\/index\.html$/,"").replace(/\/console$/,"").replace(/\/+$/,"");
const API=base+"/api/v1";
$("apibase").value=API;
function fillSelect(id,ids,optional){
  const sel=$(id);if(!sel)return;
  const cur=sel.value;
  let html=optional?"<option value=''></option>":"";
  html+=ids.map(x=>"<option>"+esc(x)+"</option>").join("");
  sel.innerHTML=html;
  if(cur && (ids.includes(cur)||cur===""))sel.value=cur;
  else if(!optional && ids.length)sel.value=ids[0];
}
async function populateTenants(){
  try{
    const d=await api("GET","/tenants");
    const sel=$("tenant");
    let saved=null;try{saved=localStorage.getItem("weft.tenant")}catch(e){}
    sel.innerHTML=d.tenants.map(t=>"<option>"+esc(t)+"</option>").join("");
    if(saved && d.tenants.includes(saved))sel.value=saved;
    else if(!d.tenants.length)toast("no tenants yet - register an agent or create a session",true);
    return d.tenants;
  }catch(e){toast(e.message,true);return []}
}
async function populateAgents(){
  if(!$("tenant").value)return [];
  try{
    const d=await api("GET","/agents");
    const ids=d.map(a=>a.id);
    fillSelect("ragent",ids,true);
    fillSelect("sa",ids,true);
    fillSelect("hagent",ids,false);
    fillSelect("hfrom",ids,false);
    fillSelect("hto",ids,false);
    return d;
  }catch(e){toast(e.message,true);return []}
}
async function populateSessions(){
  if(!$("tenant").value)return;
  try{
    const d=await api("GET","/sessions?limit=50");
    const sel=$("slist");const cur=sel.value;
    sel.innerHTML="<option value=''></option>"+d.map(s=>"<option value='"+esc(s.id)+"'>"+esc(String(s.id).slice(-8))+" &middot; "+esc(s.user_id)+" &middot; "+esc(s.status)+" &middot; "+s.message_count+"msg</option>").join("");
    if(cur)sel.value=cur;
  }catch(e){}
}
async function refreshAll(){
  await populateTenants();
  await populateAgents();
  await populateSessions();
}
$("tenant").addEventListener("change",()=>{
  try{localStorage.setItem("weft.tenant",$("tenant").value)}catch(e){}
  populateAgents();populateSessions();
});

function toast(msg,err){
  const d=document.createElement("div");
  d.className="toast"+(err?" err":"");
  d.textContent=msg;
  $("toast").appendChild(d);
  setTimeout(()=>d.remove(),err?7000:3200);
}
async function api(method,p,body){
  const t=$("tenant").value.trim();
  const h={};
  if(t)h["X-Tenant-Id"]=t;
  else if(p!=="/tenants")throw new Error("no tenant selected - create or register one first");
  let payload;
  if(body!==undefined){h["content-type"]="application/json";payload=typeof body==="string"?body:JSON.stringify(body);}
  const r=await fetch(API+p,{method,headers:h,body:payload});
  const t=await r.text();
  let d;try{d=t?JSON.parse(t):null}catch(e){d=t}
  if(!r.ok){
    let m=(d&&d.detail)?(typeof d.detail==="string"?d.detail:JSON.stringify(d.detail)):(r.status+" "+t.slice(0,300));
    throw new Error(m);
  }
  return d;
}
function bars(sc){
  if(!sc)return "";
  return Object.keys(sc).map(k=>{
    const v=Number(sc[k])||0;
    const pct=Math.max(0,Math.min(100,Math.round(v*100)));
    return "<div class='kv'><b>"+esc(k)+"</b><div><div class='bar'><i style='width:"+pct+"%'></i></div></div></div>";
  }).join("");
}

async function doRecall(){
  try{
    const b={query:$("rq").value,k:parseInt($("rk").value||"8",10)||8};
    if($("ruser").value.trim())b.user_id=$("ruser").value.trim();
    if($("ragent").value.trim())b.agent_id=$("ragent").value.trim();
    const d=await api("POST","/recall",b);
    if(!d.items.length){$("rresults").innerHTML="<p class='hint'>No results.</p>";}
    else{
      $("rresults").innerHTML=d.items.map(it=>{
        const ex=it.explanation||{};
        const paths=ex.paths||[];
        const chips=(it.matched_via||[]).map(m=>"<span class='chip'>"+esc(m)+"</span>").join("");
        return "<div class='item'>"
          +"<span class='badge "+esc(it.kind)+"'>"+esc(it.kind)+"</span>"
          +"<span class='score'>"+Number(it.score).toFixed(3)+"</span>"
          +"<div style='margin-top:4px'>"+chips+"</div>"
          +"<div class='text'>"+esc(it.text)+"</div>"
          +"<details style='margin-top:8px'><summary class='hint'>explanation &middot; "+Math.round(d.took_ms)+"ms</summary>"
          +"<div class='kv' style='margin-top:8px'>"+bars(ex.score_contributions)+"</div>"
          +(paths.length?"<div class='hint' style='margin-top:8px'>traversal</div>"+paths.map(x=>"<div class='chip'>"+esc(x)+"</div>").join(""):"")
          +"</details></div>";
      }).join("");
    }
    toast("recall: "+d.items.length+" items in "+Math.round(d.took_ms)+"ms");
  }catch(e){toast(e.message,true)}
}

function renderSession(d){
  $("sinfo").innerHTML="<div class='kv'>"
    +"<b>id</b><span>"+esc(d.id)+"</span>"
    +"<b>user_id</b><span>"+esc(d.user_id)+"</span>"
    +"<b>agent_id</b><span>"+esc(d.agent_id||"-")+"</span>"
    +"<b>status</b><span>"+esc(d.status)+"</span>"
    +"<b>messages</b><span>"+d.message_count+"</span>"
    +"<b>started</b><span>"+esc(d.started_at)+"</span></div>";
}
async function createSession(){
  try{
    const b={user_id:$("su").value};
    if($("sa").value)b.agent_id=$("sa").value;
    if($("st").value)b.title=$("st").value;
    const d=await api("POST","/sessions",b);
    $("sid").value=d.id;renderSession(d);toast("session created: "+d.id);
    populateSessions();populateAgents();
  }catch(e){toast(e.message,true)}
}
async function loadSession(){
  try{renderSession(await api("GET","/sessions/"+encodeURIComponent($("sid").value)));toast("session loaded")}
  catch(e){toast(e.message,true)}
}
async function appendMsgs(){
  try{
    const arr=JSON.parse($("smsg").value);
    const out=await api("POST","/sessions/"+encodeURIComponent($("sid").value)+"/messages",arr);
    toast("appended "+out.length+" messages");loadMessages();
  }catch(e){toast(e.message,true)}
}
async function loadMessages(){
  try{
    const out=await api("GET","/sessions/"+encodeURIComponent($("sid").value)+"/messages?limit=100");
    $("sout").innerHTML=out.map(m=>"<div class='msg "+esc(m.role)+"'><div class='meta'>#"+m.seq+" &middot; "+esc(m.role)+" &middot; "+esc(m.ts)+"</div><div>"+esc(m.content)+"</div></div>").join("");
    toast("loaded "+out.length+" messages");
  }catch(e){toast(e.message,true)}
}
async function loadTimeline(){
  try{const d=await api("GET","/sessions/"+encodeURIComponent($("sid").value)+"/timeline");$("sout").innerHTML="<pre>"+esc(JSON.stringify(d,null,2))+"</pre>"}
  catch(e){toast(e.message,true)}
}
async function consolidate(){
  try{const d=await api("POST","/sessions/"+encodeURIComponent($("sid").value)+"/consolidate");toast("consolidated: "+JSON.stringify(d))}
  catch(e){toast(e.message,true)}
}

async function loadAgents(){
  try{
    const d=await api("GET","/agents");
    $("hout").innerHTML="<div class='card'><h3>Agents</h3><table><tr><th>id</th><th>name</th><th>capabilities</th><th>sent</th><th>received</th></tr>"
      +d.map(a=>"<tr><td>"+esc(a.id)+"</td><td>"+esc(a.name)+"</td><td>"+esc((a.capabilities||[]).join(", "))+"</td><td>"+a.handoffs_sent+"</td><td>"+a.handoffs_received+"</td></tr>").join("")
      +"</table></div>";
  }catch(e){toast(e.message,true)}
}
async function loadInbox(){
  try{
    const st=$("hstatus").value.trim();
    const d=await api("GET","/agents/"+encodeURIComponent($("hagent").value)+"/inbox"+(st?"?status="+encodeURIComponent(st):""));
    if(!d.length){$("hout").innerHTML="<p class='hint'>Inbox empty.</p>";return}
    $("hout").innerHTML="<div class='card'><h3>Inbox for "+esc($("hagent").value)+"</h3><table><tr><th>status</th><th>from</th><th>title</th><th>reason</th><th>priority</th><th></th></tr>"
      +d.map(h=>{
        let a="";
        if(h.status==="pending")a="<button onclick=\"hact('"+h.id+"','claim')\">Claim</button>";
        if(h.status==="claimed")a="<button onclick=\"hact('"+h.id+"','complete')\">Complete</button> <button onclick=\"hact('"+h.id+"','reject')\">Reject</button>";
        return "<tr><td><span class='hstat'>"+esc(h.status)+"</span></td><td>"+esc(h.from_agent)+"</td><td>"+esc(h.title)+"</td><td>"+esc(h.reason)+"</td><td>"+h.priority+"</td><td>"+a+"</td></tr>";
      }).join("")
      +"</table></div>";
    toast("inbox: "+d.length+" handoffs");
  }catch(e){toast(e.message,true)}
}
async function createHandoff(){
  try{
    const d=await api("POST","/handoffs",{from_agent:$("hfrom").value,to_agent:$("hto").value,title:$("htitle").value,reason:"created from console"});
    toast("handoff created: "+d.id);loadInbox();
  }catch(e){toast(e.message,true)}
}
async function hact(id,act){
  try{
    const a=encodeURIComponent($("hagent").value);
    if(act==="claim")await api("POST","/handoffs/"+id+"/claim?agent_id="+a);
    else if(act==="complete")await api("POST","/handoffs/"+id+"/complete?agent_id="+a,{result:"confirmed via console"});
    else await api("POST","/handoffs/"+id+"/reject?agent_id="+a+"&reason="+encodeURIComponent("handled in console"));
    toast(act+" ok");loadInbox();
  }catch(e){toast(e.message,true)}
}

async function loadScope(){
  try{
    const raw=$("bscopes").value.trim();
    const d = raw
      ? await api("POST","/state/scope",raw.split(",").map(s=>s.trim()).filter(Boolean))
      : await api("GET","/state?limit=500");
    if(!d.length){$("bout").innerHTML="<p class='hint'>No state entries.</p>";return}
    $("bout").innerHTML="<div class='card'><h3>Shared state</h3><table><tr><th>scope</th><th>key</th><th>ver</th><th>updated_by</th><th>value</th></tr>"
      +d.map(e=>"<tr><td>"+esc(e.scope)+"</td><td>"+esc(e.key)+"</td><td>"+e.version+"</td><td>"+esc(e.updated_by||"-")+"</td><td><code>"+esc(JSON.stringify(e.value))+"</code></td></tr>").join("")
      +"</table></div>";
    toast("loaded "+d.length+" entries");
  }catch(e){toast(e.message,true)}
}
async function setState(){
  try{
    let val;try{val=JSON.parse($("bval").value)}catch(e){val=$("bval").value}
    const b={value:val};
    if($("bver").value.trim())b.expected_version=parseInt($("bver").value,10);
    const d=await api("POST","/state/"+encodeURIComponent($("bscope").value)+"/"+encodeURIComponent($("bkey").value),b);
    $("bver").value=d.version;toast("state set: "+d.scope+"/"+d.key+" v"+d.version);loadScope();
  }catch(e){toast(e.message,true)}
}

async function loadStats(){
  try{
    const s=await api("GET","/stats");
    const rows=Object.keys(s.counts).sort((a,b)=>s.counts[b]-s.counts[a])
      .map(k=>"<tr><td>"+esc(k)+"</td><td>"+s.counts[k]+"</td></tr>").join("");
    $("stout").innerHTML=
      "<div class='card'><div class='kv'><b>tenant</b><span>"+esc(s.tenant)+"</span><b>graphs</b><span>"+s.graphs+"</span></div></div>"
      +"<div class='card'><h3>Node counts</h3><table><tr><th>label</th><th>count</th></tr>"+rows+"</table></div>"
      +"<div class='card'><h3>Job queue</h3><pre>"+esc(JSON.stringify(s.job_queue,null,2))+"</pre></div>";
  }catch(e){toast(e.message,true)}
}
async function loadTenants(){
  try{
    const d=await api("GET","/tenants");
    $("stout").innerHTML="<div class='card'><h3>Tenants</h3>"
      +d.tenants.map(x=>"<span class='chip' style='cursor:pointer' onclick=\"setTenant('"+esc(x)+"')\">"+esc(x)+"</span>").join("")
      +"</div><p class='hint'>Click a tenant to switch the console to it.</p>";
  }catch(e){toast(e.message,true)}
}
function setTenant(t){$("tenant").value=t;try{localStorage.setItem("weft.tenant",t)}catch(e){}toast("tenant -> "+t)}

document.querySelectorAll("#tabs button").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#tabs button").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(x=>x.classList.remove("active"));
  b.classList.add("active");
  $("p-"+b.dataset.t).classList.add("active");
}));
$("rq").addEventListener("keydown",e=>{if(e.key==="Enter")doRecall()});
refreshAll();
</script>
</body>
</html>
"""

__all__ = ["INDEX_HTML"]
