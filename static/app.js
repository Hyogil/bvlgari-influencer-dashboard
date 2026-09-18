const $ = (id) => document.getElementById(id);
let current = null;

function fmtCompact(n){
  n = Number(n || 0);
  if(n >= 1_000_000) return (n/1_000_000).toFixed(n>=10_000_000?0:1).replace('.0','')+'M';
  if(n >= 1_000) return (n/1_000).toFixed(n>=100_000?0:1).replace('.0','')+'K';
  return String(Math.round(n));
}
function fmtSigned(n, digits=3){ return `${n>=0?'+':''}${Number(n).toFixed(digits)}`; }
function safeName(name, handle){
  if(!name || name === 'nan') return handle;
  const odd = (String(name).match(/[\x80-\x9f]/g)||[]).length;
  return odd >= 2 ? handle : name;
}
function hashCode(text){
  let h = 0;
  for(let i=0;i<text.length;i++) h = ((h<<5)-h) + text.charCodeAt(i);
  return Math.abs(h);
}
function avatarFileName(handle){
  return String(handle || '').replace(/^@/, '').replace(/[^a-zA-Z0-9._-]/g, '_') + '.jpg';
}
function avatarUrl(handle, platform){
  const p = String(platform || '').toLowerCase();
  const folder = p.includes('instagram') ? 'instagram'
               : p.includes('tiktok') ? 'tiktok'
               : p.includes('youtube') ? 'youtube'
               : 'sample';
  return `/Resource/images/${folder}/${encodeURIComponent(avatarFileName(handle))}`;
}
function sampleAvatarUrl(handle){
  const idx = (hashCode(String(handle)) % 6) + 1;
  return `/Resource/images/sample/sample-${idx}.svg`;
}
function avatarFallback(img, handle){
  if(img.dataset.fallbackApplied === '1') return;
  img.dataset.fallbackApplied = '1';
  img.src = sampleAvatarUrl(handle);
}
function platformIcon(platform){
  const p = String(platform || '').toLowerCase();
  if(p.includes('instagram')) return '<i class="fa-brands fa-instagram"></i>';
  if(p.includes('tiktok')) return '<i class="fa-brands fa-tiktok"></i>';
  if(p.includes('youtube')) return '<i class="fa-brands fa-youtube"></i>';
  return '<i class="fa-solid fa-user"></i>';
}
function showToast(text){
  const t=$('toast');
  if(!t) return;
  t.textContent=text;
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2600);
}
function fillSelect(id, values){ $(id).innerHTML = values.map(v=>`<option>${escapeHtml(v)}</option>`).join(''); }

function getControlValues(){
  return {
    brand_weight: Number($('brandWeight').value)/100,
    min_followers: Number($('minFollowers').value),
    min_engagement: Number($('minEngagement').value),
    suitable_threshold: Number($('suitableThreshold').value)/100,
    tree_depth: Number($('treeDepth').value),
  };
}


function updateControlLabels(){
  const brand=Number($('brandWeight').value);
  $('brandWeightLabel').textContent=`${brand}%`;
  $('campaignWeightLabel').textContent=`${100-brand}%`;
  $('minFollowersLabel').textContent=fmtCompact(Number($('minFollowers').value));
  $('minEngagementLabel').textContent=`${Number($('minEngagement').value).toFixed(1)}%`;
  $('suitableThresholdLabel').textContent=`${Number($('suitableThreshold').value)}%`;
  $('treeDepthLabel').textContent=String($('treeDepth').value);
}

function bindControls(){
  ['brandWeight','minFollowers','minEngagement','suitableThreshold','treeDepth'].forEach(id=>{
    $(id).addEventListener('input',updateControlLabels);
    $(id).addEventListener('change',()=>runAnalysis(null));
  });
}

async function init(){
  const r = await fetch('/api/options');
  if(!r.ok) throw new Error('Could not load dashboard options.');
  const o = await r.json();
  fillSelect('brand', o.brands);
  fillSelect('country', o.countries);
  fillSelect('campaign', o.campaigns);
  fillSelect('platform', o.platforms);
  $('brand').value='BVLGARI';
  $('country').value='KR';
  $('campaign').value='Luxury / Fashion';
  $('platform').value='Instagram';

  const d=o.defaults || {};
  $('brandWeight').value=String(Math.round(Number(d.brand_weight ?? .5)*100));
  $('minFollowers').value=String(Number(d.min_followers ?? 0));
  $('minEngagement').value=String(Number(d.min_engagement ?? 0));
  $('suitableThreshold').value=String(Math.round(Number(d.suitable_threshold ?? .5)*100));
  $('treeDepth').value=String(Number(d.tree_depth ?? 4));

  updateControlLabels();
  bindControls();
  $('runBtn').addEventListener('click',()=>runAnalysis(null));
  await runAnalysis();
}

async function runAnalysis(selectedHandle=null){
  const btn=$('runBtn');
  btn.disabled=true;
  btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> Analyzing…';
  try{
    const c=getControlValues();
    const p = new URLSearchParams({
      brand:$('brand').value,
      country:$('country').value,
      campaign:$('campaign').value,
      platform:$('platform').value,
      brand_weight:String(c.brand_weight),
      min_followers:String(c.min_followers),
      min_engagement:String(c.min_engagement),
      suitable_threshold:String(c.suitable_threshold),
      tree_depth:String(c.tree_depth),
    });
    if(selectedHandle) p.set('selected_handle',selectedHandle);
    const r=await fetch('/api/analyze?'+p.toString());
    if(!r.ok) throw new Error((await r.json()).detail || 'Analysis failed');
    current=await r.json();
    renderAll();
  }catch(e){
    showToast(e.message);
  } finally {
    btn.disabled=false;
    btn.innerHTML='<i class="fa-solid fa-play"></i> Run analysis';
  }
}

function renderAll(){
  $('sampleSize').textContent = current.sample_size.toLocaleString();
  $('trainingSize').textContent = current.training_size.toLocaleString();
  $('datasetSize').textContent = current.dataset_size.toLocaleString();
  $('repairedRows').textContent = current.repaired_rows.toLocaleString();
  $('contextLabel').textContent = `${current.context.brand} · ${current.context.country} · ${current.context.platform} · ${current.context.campaign}`;
  $('fitMethod').textContent = `${Math.round(current.context.brand_weight*100)}% Brand + ${Math.round(current.context.campaign_weight*100)}% Campaign`;
  renderTop3();
  renderRanking();
  renderLogisticExplanation();
  renderDecisionPath();
  renderSelected();
  renderTree();
}

function renderTop3(){
  const sel=current.selected.handle;
  $('top3').innerHTML=current.top10.slice(0,3).map(c=>`
    <article class="creator-card ${c.handle===sel?'active':''}" data-handle="${escapeAttr(c.handle)}">
      <div class="rank-pill">Top ${c.rank}</div>
      <div class="creator-main">
        <div class="avatar-wrap">
          <img class="avatar-photo" src="${avatarUrl(c.handle,c.platform)}" alt="${escapeAttr(c.handle)} profile" onerror="avatarFallback(this, '${escapeAttr(c.handle)}')">
          <span class="platform-dot">${platformIcon(c.platform)}</span>
        </div>
        <div class="creator-text">
          <div class="handle">${escapeHtml(c.handle)}</div>
          <div class="creator-name">${escapeHtml(safeName(c.name,c.handle))}</div>
          <div class="niche">
            <span class="pill-inline"><i class="fa-solid fa-flag"></i>${escapeHtml(c.country)}</span>
            <span class="pill-inline"><i class="fa-solid fa-tag"></i>${escapeHtml(c.niche)}</span>
            <span class="pill-inline">${platformIcon(c.platform)}${escapeHtml(c.platform)}</span>
          </div>
        </div>
        <div class="prob"><small>Predicted Success Probability</small><strong>${(c.selection_probability*100).toFixed(1)}%</strong><em>Logistic Regression</em></div>
      </div>
      <div class="metrics">
        <div class="metric"><strong>${fmtCompact(c.followers)}</strong><span>Followers</span></div>
        <div class="metric"><strong>${c.engagement_rate.toFixed(1)}%${c.engagement_imputed?'*':''}</strong><span>Engagement${c.engagement_imputed?' (imputed)':''}</span></div>
        <div class="metric"><strong>${(c.content_fit*100).toFixed(1)}</strong><span>Content Fit /100</span></div>
      </div>
    </article>`).join('');
  document.querySelectorAll('.creator-card').forEach(el=>el.addEventListener('click',()=>runAnalysis(el.dataset.handle)));
}

function renderRanking(){
  const selectedHandle=current.selected.handle;
  $('ranking').innerHTML=current.top10.map((c,i)=>`
    <button type="button" class="rank-row ${c.handle===selectedHandle?'active':''}" data-handle="${escapeAttr(c.handle)}" aria-label="Inspect ${escapeAttr(c.handle)}">
      <div class="rank-num">${i+1}</div>
      <div class="rank-info">
        <img class="rank-avatar" src="${avatarUrl(c.handle,c.platform)}" alt="${escapeAttr(c.handle)} profile" onerror="avatarFallback(this, '${escapeAttr(c.handle)}')">
        <div>
          <div class="rank-handle">${escapeHtml(c.handle)}</div>
          <div class="rank-sub">${escapeHtml(safeName(c.name,c.handle))} · ${escapeHtml(c.niche)} · ${escapeHtml(c.platform)} <span class="model-inline">· Logistic ${(c.selection_probability*100).toFixed(1)}%</span></div>
        </div>
      </div>
      <div class="bar-bg"><div class="bar" style="width:${Math.max(2,c.selection_probability*100)}%"></div></div>
      <div class="rank-pct">${(c.selection_probability*100).toFixed(1)}%</div>
    </button>`).join('');

  document.querySelectorAll('.rank-row').forEach(el=>{
    el.addEventListener('click', async ()=>{
      const handle=el.dataset.handle;
      if(!handle || handle===current.selected.handle) return;
      await runAnalysis(handle);
    });
  });
}

function renderLogisticExplanation(){
  const e=current.logistic_explanation;
  const s=current.selected;
  $('formulaSelected').textContent=`#${s.rank} ${s.handle}`;
  $('featureValues').innerHTML=e.features.map(f=>`
    <div class="mini-row">
      <span>${escapeHtml(f.feature_label)}<div class="subvalue">Z = ${Number(f.standardized).toFixed(3)}</div></span>
      <strong>${escapeHtml(f.raw_label)}</strong>
    </div>`).join('');

  $('contributions').innerHTML=`
    <div class="mini-row"><span>Intercept β₀</span><strong>${fmtSigned(e.intercept)}</strong></div>
    ${e.features.map(f=>`
      <div class="mini-row">
        <span>${escapeHtml(f.feature_label)}<div class="subvalue">β ${fmtSigned(f.coefficient)} × Z ${Number(f.standardized).toFixed(3)}</div></span>
        <strong>${fmtSigned(f.contribution)}</strong>
      </div>`).join('')}
    <div class="mini-row total"><span>Total z</span><strong>${Number(e.logit).toFixed(3)}</strong></div>
    <div class="mini-row total"><span>Probability</span><strong>${(e.probability*100).toFixed(2)}%</strong></div>`;
  renderSigmoid(e.logit,e.probability);
}

function renderSigmoid(logit, probability){
  const svg=$('sigmoidSvg');
  svg.innerHTML='';
  const NS='http://www.w3.org/2000/svg';
  const W=300,H=205,L=36,R=12,T=20,B=32;
  const xmin=-6,xmax=6,ymin=0,ymax=1;
  const X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R);
  const Y=y=>T+(ymax-y)/(ymax-ymin)*(H-T-B);
  const line=(x1,y1,x2,y2,stroke='#2d3850',width=1,dash='')=>{
    const el=document.createElementNS(NS,'line');el.setAttribute('x1',x1);el.setAttribute('y1',y1);el.setAttribute('x2',x2);el.setAttribute('y2',y2);el.setAttribute('stroke',stroke);el.setAttribute('stroke-width',width);if(dash)el.setAttribute('stroke-dasharray',dash);svg.appendChild(el);return el;
  };
  const text=(x,y,value,size=9,fill='#91a0ba',anchor='middle',weight='400')=>{
    const el=document.createElementNS(NS,'text');el.setAttribute('x',x);el.setAttribute('y',y);el.setAttribute('font-size',size);el.setAttribute('fill',fill);el.setAttribute('text-anchor',anchor);el.setAttribute('font-weight',weight);el.textContent=value;svg.appendChild(el);return el;
  };
  [0,.25,.5,.75,1].forEach(v=>{line(L,Y(v),W-R,Y(v),'#253045');text(L-6,Y(v)+3,v.toFixed(v===0||v===1?0:2),8,'#8a97ae','end');});
  [-6,-4,-2,0,2,4,6].forEach(v=>{line(X(v),T,X(v),H-B,'#1d2637');text(X(v),H-17,String(v),8);});
  line(L,T,L,H-B,'#54617a',1.1);line(L,H-B,W-R,H-B,'#54617a',1.1);
  let d='';
  for(let i=0;i<=120;i++){
    const x=xmin+(xmax-xmin)*i/120;
    const p=1/(1+Math.exp(-x));
    d+=(i===0?'M':'L')+`${X(x).toFixed(2)} ${Y(p).toFixed(2)} `;
  }
  const path=document.createElementNS(NS,'path');path.setAttribute('d',d);path.setAttribute('fill','none');path.setAttribute('stroke','#d0a15f');path.setAttribute('stroke-width','3');svg.appendChild(path);
  const px=Math.max(xmin,Math.min(xmax,logit));
  line(X(px),Y(probability),X(px),H-B,'#c86f7a',1,'4 3');
  line(L,Y(probability),X(px),Y(probability),'#7d5cff',1,'3 3');
  const dot=document.createElementNS(NS,'circle');dot.setAttribute('cx',X(px));dot.setAttribute('cy',Y(probability));dot.setAttribute('r','5');dot.setAttribute('fill','#ec5d73');dot.setAttribute('stroke','#fff');dot.setAttribute('stroke-width','2');svg.appendChild(dot);
  text(X(px),Math.max(12,Y(probability)-11),`z=${Number(logit).toFixed(3)}`,9,'#f1d2a2','middle','700');
  text(W-12,13,`${(probability*100).toFixed(2)}%`,12,'#f1d2a2','end','800');
  text((L+W-R)/2,H-4,'Logit z',9,'#91a0ba');
  const ylabel=text(10,(T+H-B)/2,'P(Y=1)',9,'#91a0ba');ylabel.setAttribute('transform',`rotate(-90 10 ${(T+H-B)/2})`);
  $('sigmoidCaption').textContent=`Sigmoid(${Number(logit).toFixed(3)}) = ${(probability*100).toFixed(2)}%`;
}

function renderDecisionPath(){
  const s=current.selected;
  $('decisionPath').innerHTML=s.decision_path.map((p,i)=>`
    <div class="path-step"><span class="path-num">${i+1}</span><div><b>${escapeHtml(p.feature_label)}</b> ${p.operator} ${escapeHtml(p.threshold_label)}<br><span style="color:#a3afc4">Actual: ${escapeHtml(p.actual_label)}</span></div></div>`).join('')+
    `<div class="result">→ ${escapeHtml(leafLabelForSelected())} · Tree P ${(s.tree_probability*100).toFixed(1)}% · threshold ${(current.context.suitable_threshold*100).toFixed(0)}%</div>`;
}
function leafLabelForSelected(){
  const leaf=findNode(current.tree,current.selected.leaf_id);
  return leaf?.label || 'Result';
}

function renderSelected(){
  const s=current.selected;
  const status=s.tree_suitable ? '<span class="tree-status good">Suitable</span>' : '<span class="tree-status bad">Not Suitable</span>';
  $('selectedInfo').innerHTML=`
    <div class="selected-head">
      <img class="selected-avatar" src="${avatarUrl(s.handle,s.platform)}" alt="Profile for ${escapeAttr(s.handle)}" onerror="avatarFallback(this, '${escapeAttr(s.handle)}')">
      <div>
        <div class="selected-name">${escapeHtml(s.handle)} ${status}</div>
        <div class="selected-display-name">${escapeHtml(safeName(s.name,s.handle))}</div>
        <div class="selected-meta">${escapeHtml(s.country)} · ${escapeHtml(s.platform)} · ${escapeHtml(s.niche)}</div>
      </div>
    </div>
    <div class="selected-grid five">
      <div><b>${(s.selection_probability*100).toFixed(1)}%</b>Predicted Success Probability</div>
      <div><b>${s.profile_fit.toFixed(3)}</b>Profile Fit</div>
      <div><b>${s.content_fit.toFixed(3)}</b>Content Fit</div>
      <div><b>${s.campaign_history_score.toFixed(3)}</b>History Score</div>
      <div><b>${s.engagement_rate.toFixed(1)}%${s.engagement_imputed?'*':''}</b>Engagement</div>
      <div><b>${fmtCompact(s.followers)}</b>Followers</div>
      <div><b>${(s.tree_probability*100).toFixed(1)}%</b>Tree leaf probability</div>
    </div>`;
  $('takeaway').innerHTML=`Rank <b>#${s.rank}</b> is determined solely by the <b>Logistic Regression predicted probability</b>. The model uses 5 features: Profile Fit, Content Fit, Campaign History Score, Engagement Rate, and log Followers. Brand ↔ Campaign changes the engineered Profile Fit and retrains the model as a sensitivity analysis. Minimum Followers and Engagement filter candidates. Suitable Threshold changes the Tree label and Tree Depth changes Tree complexity; those Decision Tree controls do not directly change the Logistic Regression ranking. <b>Important:</b> content/history fields and the target are simulated prototype data.`;
}

function findNode(n,id){ if(n.id===id) return n; if(n.is_leaf) return null; return findNode(n.left,id)||findNode(n.right,id); }

function layoutTree(root){
  const nodes=[], edges=[];
  let leafCursor=54;
  const leafGap=106;
  const yStart=38;
  const yGap=78;
  let maxDepth=0;
  function visit(n,depth,parent=null,branch=''){
    maxDepth=Math.max(maxDepth,depth);
    let x;
    if(n.is_leaf){ x=leafCursor; leafCursor+=leafGap; }
    else{
      const lx=visit(n.left,depth+1,n,'≤');
      const rx=visit(n.right,depth+1,n,'>');
      x=(lx+rx)/2;
    }
    const y=yStart+depth*yGap;
    nodes.push({...n,x,y});
    if(parent) edges.push({from:parent.id,to:n.id,branch});
    return x;
  }
  visit(root,0);
  const width=Math.max(720,leafCursor+18);
  const height=Math.max(245,yStart+maxDepth*yGap+45);
  return {nodes,edges,width,height};
}

function renderTree(){
  const svg=$('treeSvg');svg.innerHTML='';
  const NS='http://www.w3.org/2000/svg';
  const layout=layoutTree(current.tree);const {nodes,edges}=layout;
  svg.setAttribute('viewBox',`0 0 ${layout.width} ${layout.height}`);
  svg.removeAttribute('width');svg.removeAttribute('height');
  svg.style.width='100%';svg.style.height='auto';svg.style.aspectRatio=`${layout.width} / ${layout.height}`;
  const byId=new Map(nodes.map(n=>[n.id,n]));
  const pathSet=new Set(current.selected.path_node_ids);
  const nodeHalfW=50,nodeHalfH=20;

  edges.forEach(e=>{
    const a=byId.get(e.from),b=byId.get(e.to);
    const selected=pathSet.has(a.id)&&pathSet.has(b.id);
    const line=document.createElementNS(NS,'line');
    line.setAttribute('x1',a.x);line.setAttribute('y1',a.y+nodeHalfH);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y-nodeHalfH);
    line.setAttribute('stroke',selected?'#d0a15f':'#536078');line.setAttribute('stroke-width',selected?'2.8':'1.4');svg.appendChild(line);
    const label=document.createElementNS(NS,'text');
    label.setAttribute('x',(a.x+b.x)/2);label.setAttribute('y',(a.y+b.y)/2-4);label.setAttribute('text-anchor','middle');
    label.setAttribute('font-size','9');label.setAttribute('font-weight',selected?'700':'500');label.setAttribute('fill',selected?'#f1d2a2':'#9da8bc');label.textContent=e.branch;svg.appendChild(label);
  });

  nodes.forEach(n=>{
    const selected=pathSet.has(n.id);const g=document.createElementNS(NS,'g');
    const rect=document.createElementNS(NS,'rect');
    rect.setAttribute('x',n.x-nodeHalfW);rect.setAttribute('y',n.y-nodeHalfH);rect.setAttribute('width',nodeHalfW*2);rect.setAttribute('height',nodeHalfH*2);rect.setAttribute('rx','8');
    rect.setAttribute('fill',selected?'#2b1f20':'#111a2a');rect.setAttribute('stroke',selected?'#d0a15f':'#556178');rect.setAttribute('stroke-width',selected?'2.2':'1.05');g.appendChild(rect);
    const t=document.createElementNS(NS,'text');
    t.setAttribute('x',n.x);t.setAttribute('y',n.y-6);t.setAttribute('text-anchor','middle');t.setAttribute('font-size','8.3');
    t.setAttribute('font-weight',selected?'700':'600');t.setAttribute('fill',selected?'#fff0d1':'#dfe6f5');
    const lines=n.is_leaf?[n.label,`P=${Math.round(n.positive_probability*100)}% · n=${n.samples}`]:[n.feature_label,`≤ ${n.threshold_label}? · n=${n.samples}`];
    lines.forEach((txt,i)=>{const sp=document.createElementNS(NS,'tspan');sp.setAttribute('x',n.x);sp.setAttribute('dy',i===0?'0':'12');sp.textContent=txt;t.appendChild(sp);});
    g.appendChild(t);svg.appendChild(g);
  });
}

function escapeHtml(s){ return String(s??'').replace(/[&<>'"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch])); }
function escapeAttr(s){ return escapeHtml(s); }

init().catch(e=>showToast(e.message));
