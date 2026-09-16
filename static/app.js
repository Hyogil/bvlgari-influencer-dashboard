const $ = (id) => document.getElementById(id);
let current = null;

function fmtCompact(n){
  if(n >= 1_000_000) return (n/1_000_000).toFixed(n>=10_000_000?0:1).replace('.0','')+'M';
  if(n >= 1_000) return (n/1_000).toFixed(n>=100_000?0:1).replace('.0','')+'K';
  return String(Math.round(n));
}
function fmtSigned(n, digits=3){ return `${n>=0?'+':''}${Number(n).toFixed(digits)}`; }
function initials(handle){
  return handle.replace('@','').split(/[._-]/).slice(0,2).map(x=>x[0]?.toUpperCase()||'').join('').slice(0,2) || 'IN';
}
function safeName(name, handle){
  if(!name || name === 'nan') return handle;
  const odd = (name.match(/[\x80-\x9f]/g)||[]).length;
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
  const t=$('toast'); t.textContent=text; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2400);
}

let scenarioTimer = null;
const FOLLOWER_STEPS = [0, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000];

function minFollowersValue(){
  const idx = Math.max(0, Math.min(FOLLOWER_STEPS.length-1, Number($('minFollowers')?.value || 0)));
  return FOLLOWER_STEPS[idx];
}

function scenarioValues(){
  const brandPct = Number($('brandWeight')?.value ?? 50);
  return {
    brandWeight: brandPct / 100,
    minFollowers: minFollowersValue(),
    minEngagement: Number($('minEngagement')?.value ?? 0),
    decisionThreshold: Number($('decisionThreshold')?.value ?? 50) / 100,
    treeDepth: Number($('treeDepth')?.value ?? 4),
  };
}

function updateScenarioLabels(){
  const v=scenarioValues();
  const brandPct=Math.round(v.brandWeight*100);
  if($('brandWeightLabel')) $('brandWeightLabel').textContent=`${brandPct}%`;
  if($('campaignWeightLabel')) $('campaignWeightLabel').textContent=`${100-brandPct}%`;
  if($('minFollowersLabel')) $('minFollowersLabel').textContent=v.minFollowers===0?'Any':fmtCompact(v.minFollowers);
  if($('minEngagementLabel')) $('minEngagementLabel').textContent=`${v.minEngagement.toFixed(1)}%`;
  if($('decisionThresholdLabel')) $('decisionThresholdLabel').textContent=`${Math.round(v.decisionThreshold*100)}%`;
  if($('treeDepthLabel')) $('treeDepthLabel').textContent=`Depth ${v.treeDepth}`;
  if($('fitMethod')) $('fitMethod').textContent=`${brandPct}% Brand + ${100-brandPct}% Campaign`;
}

function scheduleScenarioRun(){
  updateScenarioLabels();
  clearTimeout(scenarioTimer);
  scenarioTimer=setTimeout(()=>runAnalysis(null),280);
}

function ensureScenarioSliders(){
  if($('scenarioControls')) return;

  // Put the scenario controls in a full-width row immediately above the Top 3
  // section instead of inside the narrow Model Summary card.
  const top3 = $('top3');
  const top3Section = top3?.closest('section') || top3?.parentElement?.parentElement;
  const host = top3Section?.parentElement || document.body;
  if(!host) return;

  if(!$('scenarioControlStyles')){
    const style=document.createElement('style');
    style.id='scenarioControlStyles';
    style.textContent=`
      #scenarioControls{
        width:100%;
        box-sizing:border-box;
        margin:14px 0 18px;
        padding:16px 18px 18px;
        border:1px solid #334158;
        border-radius:15px;
        background:linear-gradient(180deg,#121b2b 0%,#0e1725 100%);
        box-shadow:0 8px 24px rgba(0,0,0,.16);
      }
      #scenarioControls .scenario-head{
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:16px;
        margin-bottom:14px;
      }
      #scenarioControls .scenario-title{
        font-size:14px;
        font-weight:800;
        color:#f3f6fb;
        letter-spacing:.1px;
      }
      #scenarioControls .scenario-help{
        font-size:11px;
        color:#91a0ba;
        white-space:nowrap;
      }
      #scenarioControls .scenario-grid{
        display:grid;
        grid-template-columns:repeat(5,minmax(175px,1fr));
        gap:14px;
      }
      #scenarioControls .scenario-card{
        min-width:0;
        padding:12px 13px 13px;
        border:1px solid #28354a;
        border-radius:12px;
        background:#101a2a;
      }
      #scenarioControls .scenario-label{
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:10px;
        margin-bottom:10px;
        color:#cdd6e6;
        font-size:11px;
        line-height:1.25;
      }
      #scenarioControls .scenario-label strong{
        color:#f1c678;
        font-size:13px;
        font-weight:800;
        white-space:nowrap;
      }
      #scenarioControls .weight-pair{
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:12px;
        margin-bottom:10px;
        color:#cdd6e6;
        font-size:11px;
      }
      #scenarioControls .weight-pair b{
        color:#f1c678;
        font-size:13px;
      }
      #scenarioControls input[type="range"]{
        -webkit-appearance:none;
        appearance:none;
        width:100%;
        height:24px;
        margin:0;
        padding:0;
        background:transparent;
        cursor:pointer;
      }
      #scenarioControls input[type="range"]::-webkit-slider-runnable-track{
        height:7px;
        border-radius:999px;
        background:linear-gradient(90deg,#d7a358,#b56a77 52%,#765cff);
        border:1px solid rgba(255,255,255,.08);
      }
      #scenarioControls input[type="range"]::-webkit-slider-thumb{
        -webkit-appearance:none;
        appearance:none;
        width:20px;
        height:20px;
        margin-top:-7px;
        border-radius:50%;
        background:#fff4de;
        border:3px solid #d49b50;
        box-shadow:0 0 0 3px rgba(212,155,80,.16),0 2px 7px rgba(0,0,0,.5);
      }
      #scenarioControls input[type="range"]::-moz-range-track{
        height:7px;
        border-radius:999px;
        background:linear-gradient(90deg,#d7a358,#b56a77 52%,#765cff);
        border:1px solid rgba(255,255,255,.08);
      }
      #scenarioControls input[type="range"]::-moz-range-thumb{
        width:18px;
        height:18px;
        border-radius:50%;
        background:#fff4de;
        border:3px solid #d49b50;
        box-shadow:0 0 0 3px rgba(212,155,80,.16),0 2px 7px rgba(0,0,0,.5);
      }
      #scenarioControls .range-hint{
        display:flex;
        justify-content:space-between;
        margin-top:5px;
        color:#66758e;
        font-size:9px;
      }
      @media (max-width:1250px){
        #scenarioControls .scenario-grid{grid-template-columns:repeat(3,minmax(190px,1fr));}
      }
      @media (max-width:820px){
        #scenarioControls .scenario-grid{grid-template-columns:1fr;}
        #scenarioControls .scenario-head{align-items:flex-start;flex-direction:column;}
        #scenarioControls .scenario-help{white-space:normal;}
      }
    `;
    document.head.appendChild(style);
  }

  const wrap=document.createElement('section');
  wrap.id='scenarioControls';
  wrap.innerHTML=`
    <div class="scenario-head">
      <div>
        <div class="scenario-title"><i class="fa-solid fa-sliders"></i> Scenario & Model Controls</div>
      </div>
      <div class="scenario-help">Move a slider to re-run the analysis automatically</div>
    </div>

    <div class="scenario-grid">
      <div class="scenario-card">
        <div class="weight-pair">
          <span>Brand <b id="brandWeightLabel">50%</b></span>
          <span>Campaign <b id="campaignWeightLabel">50%</b></span>
        </div>
        <input id="brandWeight" type="range" min="0" max="100" step="5" value="50">
        <div class="range-hint"><span>Brand-driven</span><span>Balanced</span><span>Campaign-driven</span></div>
      </div>

      <div class="scenario-card">
        <div class="scenario-label">
          <span>Minimum Followers</span><strong id="minFollowersLabel">Any</strong>
        </div>
        <input id="minFollowers" type="range" min="0" max="7" step="1" value="0">
        <div class="range-hint"><span>Any</span><span>500K</span><span>10M</span></div>
      </div>

      <div class="scenario-card">
        <div class="scenario-label">
          <span>Minimum Engagement</span><strong id="minEngagementLabel">0.0%</strong>
        </div>
        <input id="minEngagement" type="range" min="0" max="10" step="0.5" value="0">
        <div class="range-hint"><span>0%</span><span>5%</span><span>10%</span></div>
      </div>

      <div class="scenario-card">
        <div class="scenario-label">
          <span>Suitable Threshold</span><strong id="decisionThresholdLabel">50%</strong>
        </div>
        <input id="decisionThreshold" type="range" min="30" max="80" step="5" value="50">
        <div class="range-hint"><span>30%</span><span>55%</span><span>80%</span></div>
      </div>

      <div class="scenario-card">
        <div class="scenario-label">
          <span>Decision Tree Complexity</span><strong id="treeDepthLabel">Depth 4</strong>
        </div>
        <input id="treeDepth" type="range" min="2" max="6" step="1" value="4">
        <div class="range-hint"><span>Simple · 2</span><span>4</span><span>Complex · 6</span></div>
      </div>
    </div>`;

  if(top3Section && top3Section.parentElement===host){
    host.insertBefore(wrap, top3Section);
  }else{
    host.prepend(wrap);
  }

  ['brandWeight','minFollowers','minEngagement','decisionThreshold','treeDepth'].forEach(id=>{
    $(id).addEventListener('input', scheduleScenarioRun);
  });
  updateScenarioLabels();
}

async function init(){
  const r = await fetch('/api/options');
  if(!r.ok) throw new Error('Could not load dashboard options.');
  const o = await r.json();
  fillSelect('brand', o.brands); fillSelect('country', o.countries); fillSelect('campaign', o.campaigns); fillSelect('platform', o.platforms);
  $('brand').value='BVLGARI'; $('country').value='KR'; $('campaign').value='Luxury / Fashion'; $('platform').value='Instagram';
  $('runBtn').addEventListener('click',()=>runAnalysis());
  ensureScenarioSliders();
  await runAnalysis();
}
function fillSelect(id, values){ $(id).innerHTML = values.map(v=>`<option>${escapeHtml(v)}</option>`).join(''); }

async function runAnalysis(selectedHandle=null){
  const btn=$('runBtn'); btn.disabled=true; btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> Analyzing…';
  try{
    const sv=scenarioValues();
    const p = new URLSearchParams({
      brand:$('brand').value,
      country:$('country').value,
      campaign:$('campaign').value,
      platform:$('platform').value,
      brand_weight:String(sv.brandWeight),
      min_followers:String(sv.minFollowers),
      min_engagement:String(sv.minEngagement),
      decision_threshold:String(sv.decisionThreshold),
      tree_depth:String(sv.treeDepth)
    });
    if(selectedHandle) p.set('selected_handle',selectedHandle);
    const r=await fetch('/api/analyze?'+p.toString());
    if(!r.ok) throw new Error((await r.json()).detail || 'Analysis failed');
    current=await r.json(); renderAll();
  }catch(e){ showToast(e.message); }
  finally{ btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-play"></i> Run analysis'; }
}

function renderAll(){
  $('sampleSize').textContent = current.sample_size.toLocaleString();
  $('trainingSize').textContent = current.training_size.toLocaleString();
  $('datasetSize').textContent = current.dataset_size.toLocaleString();
  $('repairedRows').textContent = current.repaired_rows.toLocaleString();
  $('contextLabel').textContent = `${current.context.brand} · ${current.context.country} · ${current.context.platform} · ${current.context.campaign}`;
  if($('fitMethod')) $('fitMethod').textContent = current.fit_method?.label || '50% Brand + 50% Campaign';
  if(current.context){
    if($('brandWeight')) $('brandWeight').value=String(Math.round(Number(current.context.brand_weight ?? 0.5)*100));
    if($('minFollowers')){
      const target=Number(current.context.min_followers ?? 0);
      let idx=FOLLOWER_STEPS.indexOf(target);
      if(idx<0) idx=0;
      $('minFollowers').value=String(idx);
    }
    if($('minEngagement')) $('minEngagement').value=String(Number(current.context.min_engagement ?? 0));
    if($('decisionThreshold')) $('decisionThreshold').value=String(Math.round(Number(current.context.decision_threshold ?? 0.5)*100));
    if($('treeDepth')) $('treeDepth').value=String(Number(current.context.tree_depth ?? 4));
    updateScenarioLabels();
  }
  renderTop3();
  ensureDualRankingCards();
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
          <img class="avatar-photo" src="${avatarUrl(c.handle,c.platform)}" alt="${escapeAttr(c.handle)} profile" onerror="avatarFallback(this, \'${escapeAttr(c.handle)}\')">
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
        <div class="prob"><small>Selection probability</small><strong>${(c.selection_probability*100).toFixed(1)}%</strong></div>
      </div>
      <div class="metrics">
        <div class="metric"><strong>${fmtCompact(c.followers)}</strong><span>Followers</span></div>
        <div class="metric"><strong>${c.engagement_rate.toFixed(1)}%${c.engagement_imputed?'*':''}</strong><span>Engagement${c.engagement_imputed?' (imputed)':''}</span></div>
        <div class="metric"><strong>${c.verified?'Yes':'No'}${c.verified_imputed?'*':''}</strong><span>Verified${c.verified_imputed?' (imputed)':''}</span></div>
      </div>
    </article>`).join('');
  document.querySelectorAll('.creator-card').forEach(el=>el.addEventListener('click',()=>runAnalysis(el.dataset.handle)));
}

function findRankingCard(){
  const list=$('ranking');
  if(!list) return null;
  let node=list.parentElement;
  // Search only a few levels upward for the card that owns the ranking title.
  for(let i=0;i<4 && node;i++,node=node.parentElement){
    const heading=node.querySelector('h1,h2,h3,h4,.panel-title,.card-title');
    if(heading && /Top\s*10|candidates by probability/i.test(heading.textContent||'')) return node;
  }
  return list.parentElement;
}

function ensureDualRankingCards(){
  const logisticList=$('ranking');
  if(!logisticList) return;

  const logisticCard=findRankingCard();
  if(logisticCard){
    const heading=logisticCard.querySelector('h1,h2,h3,h4,.panel-title,.card-title');
    if(heading) heading.textContent='Top 10 candidates by probability (Logistic Regression)';
    const helper=[...logisticCard.querySelectorAll('p,small,.subtitle,.subtext')].find(
      el => /candidate|probability|formula|Decision Tree/i.test(el.textContent||'')
    );
    if(helper) helper.textContent='Ranked by Logistic Regression selection probability. Click a creator to inspect the same creator across both models.';
  }

  if($('treeRankingCard')) return;

  if(!$('dualRankingStyles')){
    const style=document.createElement('style');
    style.id='dualRankingStyles';
    style.textContent=`
      #treeRankingCard{
        margin-top:14px;
        padding:14px;
        border:1px solid #2d3a50;
        border-radius:15px;
        background:linear-gradient(180deg,#111a2a 0%,#0d1624 100%);
        box-sizing:border-box;
      }
      #treeRankingCard .dual-rank-title{
        margin:0 0 4px;
        color:#f3f6fb;
        font-size:14px;
        font-weight:800;
      }
      #treeRankingCard .dual-rank-sub{
        margin:0 0 12px;
        color:#91a0ba;
        font-size:10px;
        line-height:1.35;
      }
      #rankingTree{
        display:grid;
        gap:6px;
      }
      #treeRankingCard .rank-row{
        width:100%;
      }
      #treeRankingCard .tree-prob-note{
        color:#f1c678;
        font-size:9px;
        white-space:nowrap;
      }
    `;
    document.head.appendChild(style);
  }

  const card=document.createElement('section');
  card.id='treeRankingCard';
  card.innerHTML=`
    <div class="dual-rank-title"><i class="fa-solid fa-code-branch"></i> Top 10 candidates by probability (Decision Tree)</div>
    <div class="dual-rank-sub">Ranked by the positive-class probability of the Decision Tree leaf. Ties are broken by Brand Fit, Engagement, then Followers.</div>
    <div id="rankingTree"></div>
  `;

  // Keep the second ranking visually paired with the current ranking card.
  // If we can identify the original card, place it immediately after it.
  const anchor=logisticCard || logisticList.parentElement;
  if(anchor && anchor.parentElement){
    anchor.insertAdjacentElement('afterend',card);
  }else{
    logisticList.insertAdjacentElement('afterend',card);
  }
}

function rankingRowHtml(c, i, method){
  const selectedHandle=current.selected.handle;
  const isTree=method==='tree';
  const probability=isTree ? Number(c.tree_probability||0) : Number(c.selection_probability||0);
  const rank=isTree ? Number(c.tree_rank||i+1) : Number(c.rank||i+1);
  const methodText=isTree ? 'Decision Tree' : 'Logistic Regression';

  return `
    <button type="button" class="rank-row ${c.handle===selectedHandle?'active':''}" data-handle="${escapeAttr(c.handle)}" aria-label="Inspect ${escapeAttr(c.handle)}">
      <div class="rank-num">${rank}</div>
      <div class="rank-info">
        <img class="rank-avatar" src="${avatarUrl(c.handle,c.platform)}" alt="${escapeAttr(c.handle)} profile" onerror="avatarFallback(this, \'${escapeAttr(c.handle)}\')">
        <div>
          <div class="rank-handle">${escapeHtml(c.handle)}</div>
          <div class="rank-sub">${escapeHtml(safeName(c.name,c.handle))} · ${escapeHtml(c.niche)} · ${escapeHtml(c.platform)}</div>
        </div>
      </div>
      <div class="bar-bg"><div class="bar" style="width:${Math.max(2,probability*100)}%"></div></div>
      <div class="rank-pct">${(probability*100).toFixed(1)}%</div>
    </button>`;
}

function bindRankingClicks(containerSelector){
  document.querySelectorAll(`${containerSelector} .rank-row`).forEach(el=>{
    el.addEventListener('click', async ()=>{
      const handle=el.dataset.handle;
      if(!handle || handle===current.selected.handle) return;
      await runAnalysis(handle);
    });
  });
}

function renderRanking(){
  const logistic=current.top10_logistic || current.top10 || [];
  const tree=current.top10_tree || [];

  $('ranking').innerHTML=logistic.map((c,i)=>rankingRowHtml(c,i,'logistic')).join('');

  const treeContainer=$('rankingTree');
  if(treeContainer){
    treeContainer.innerHTML=tree.map((c,i)=>rankingRowHtml(c,i,'tree')).join('');
  }

  bindRankingClicks('#ranking');
  bindRankingClicks('#rankingTree');
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

  renderSigmoid(e.logit,e.probability,Number(current.context?.decision_threshold ?? 0.5));
}

function renderSigmoid(logit, probability, decisionThreshold=0.5){
  const svg=$('sigmoidSvg');
  svg.innerHTML='';
  const NS='http://www.w3.org/2000/svg';
  const W=300,H=205,L=36,R=12,T=20,B=32;
  const xmin=-6,xmax=6,ymin=0,ymax=1;
  const X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R);
  const Y=y=>T+(ymax-y)/(ymax-ymin)*(H-T-B);

  const line=(x1,y1,x2,y2,stroke='#2d3850',width=1,dash='')=>{
    const el=document.createElementNS(NS,'line'); el.setAttribute('x1',x1);el.setAttribute('y1',y1);el.setAttribute('x2',x2);el.setAttribute('y2',y2);el.setAttribute('stroke',stroke);el.setAttribute('stroke-width',width); if(dash)el.setAttribute('stroke-dasharray',dash);svg.appendChild(el); return el;
  };
  const text=(x,y,value,size=9,fill='#91a0ba',anchor='middle',weight='400')=>{
    const el=document.createElementNS(NS,'text');el.setAttribute('x',x);el.setAttribute('y',y);el.setAttribute('font-size',size);el.setAttribute('fill',fill);el.setAttribute('text-anchor',anchor);el.setAttribute('font-weight',weight);el.textContent=value;svg.appendChild(el);return el;
  };

  [0,.25,.5,.75,1].forEach(v=>{line(L,Y(v),W-R,Y(v),'#253045');text(L-6,Y(v)+3,v.toFixed(v===0||v===1?0:2),8,'#8a97ae','end');});
  [-6,-4,-2,0,2,4,6].forEach(v=>{line(X(v),T,X(v),H-B,'#1d2637');text(X(v),H-17,String(v),8);});
  line(L,T,L,H-B,'#54617a',1.1); line(L,H-B,W-R,H-B,'#54617a',1.1);
  line(L,Y(decisionThreshold),W-R,Y(decisionThreshold),'#d0a15f',1,'5 4');
  text(W-R-2,Y(decisionThreshold)-5,`threshold ${Math.round(decisionThreshold*100)}%`,8,'#d0a15f','end','600');

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
  const ylabel=text(10,(T+H-B)/2,'P(Y=1)',9,'#91a0ba'); ylabel.setAttribute('transform',`rotate(-90 10 ${(T+H-B)/2})`);
  $('sigmoidCaption').textContent=`Sigmoid(${Number(logit).toFixed(3)}) = ${(probability*100).toFixed(2)}%`;
}

function renderDecisionPath(){
  const s=current.selected;
  $('decisionPath').innerHTML=s.decision_path.map((p,i)=>`
    <div class="path-step"><span class="path-num">${i+1}</span><div><b>${escapeHtml(p.feature_label)}</b> ${p.operator} ${escapeHtml(p.threshold_label)}<br><span style="color:#a3afc4">Actual: ${escapeHtml(p.actual_label)}</span></div></div>`).join('')+
    `<div class="result">→ ${escapeHtml(leafLabelForSelected())} · Tree P(positive) ${(Number(s.tree_positive_probability||0)*100).toFixed(1)}% · threshold ${Math.round(Number(current.context?.decision_threshold ?? 0.5)*100)}%</div>`;
}
function leafLabelForSelected(){
  const leaf=findNode(current.tree,current.selected.leaf_id);
  return leaf?.label || 'Result';
}

function renderSelected(){
  const s=current.selected;
  $('selectedInfo').innerHTML=`
    <div class="selected-head">
      <img class="selected-avatar" src="${avatarUrl(s.handle,s.platform)}" alt="${escapeAttr(s.handle)} profile" onerror="avatarFallback(this, \'${escapeAttr(s.handle)}\')">
      <div>
        <div class="selected-name">${escapeHtml(s.handle)} ${s.verified?'<span class="verified-chip"><i class="fa-solid fa-badge-check"></i> Verified</span>':''}</div>
        <div class="selected-display-name">${escapeHtml(safeName(s.name,s.handle))}</div>
        <div class="selected-meta">${escapeHtml(s.country)} · ${escapeHtml(s.platform)} · ${escapeHtml(s.niche)}</div>
      </div>
    </div>
    <div class="selected-grid">
      <div><b>${(s.selection_probability*100).toFixed(1)}%</b>Logistic probability</div>
      <div><b>${s.brand_fit.toFixed(2)}</b>Brand fit</div>
      <div><b>${s.engagement_rate.toFixed(1)}%${s.engagement_imputed?'*':''}</b>Engagement${s.engagement_imputed?' (imputed)':''}</div>
      <div><b>${fmtCompact(s.followers)}</b>Followers</div>
      <div><b>${(Number(s.tree_positive_probability||0)*100).toFixed(1)}%</b>Tree positive probability</div>
    </div>`;
  const bw=Math.round(Number(current.fit_method?.brand_weight ?? 0.5)*100);
  const cw=100-bw;
  const dt=Number(current.context?.decision_threshold ?? 0.5);
  const disagreement = (s.selection_probability>=dt) !== (Number(s.tree_prediction)===1);
  const followerText=Number(current.context?.min_followers||0)>0 ? fmtCompact(Number(current.context.min_followers)) : 'Any';
  $('takeaway').innerHTML=`Rank <b>#${s.rank}</b>. Logistic Regression supplies the ranking probability; the Decision Tree is a separate rule-based approximation and can disagree${disagreement?' <b>(disagreement case)</b>':''}. Current scenario: <b>${bw}% Brand + ${cw}% Campaign</b>, minimum followers <b>${followerText}</b>, minimum engagement <b>${Number(current.context?.min_engagement||0).toFixed(1)}%</b>, suitable threshold <b>${Math.round(dt*100)}%</b>, tree depth <b>${Number(current.context?.tree_depth||4)}</b>.`;
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
  const svg=$('treeSvg'); svg.innerHTML='';
  const NS='http://www.w3.org/2000/svg';
  const layout=layoutTree(current.tree); const {nodes,edges}=layout;
  svg.setAttribute('viewBox',`0 0 ${layout.width} ${layout.height}`);
  svg.removeAttribute('width'); svg.removeAttribute('height');
  svg.style.width='100%'; svg.style.height='auto';
  svg.style.aspectRatio=`${layout.width} / ${layout.height}`;

  const byId=new Map(nodes.map(n=>[n.id,n]));
  const pathSet=new Set(current.selected.path_node_ids);
  const nodeHalfW=50,nodeHalfH=20;

  edges.forEach(e=>{
    const a=byId.get(e.from), b=byId.get(e.to);
    const selected=pathSet.has(a.id)&&pathSet.has(b.id);
    const line=document.createElementNS(NS,'line');
    line.setAttribute('x1',a.x); line.setAttribute('y1',a.y+nodeHalfH); line.setAttribute('x2',b.x); line.setAttribute('y2',b.y-nodeHalfH);
    line.setAttribute('stroke',selected?'#d0a15f':'#536078'); line.setAttribute('stroke-width',selected?'2.8':'1.4'); svg.appendChild(line);
    const label=document.createElementNS(NS,'text');
    label.setAttribute('x',(a.x+b.x)/2); label.setAttribute('y',(a.y+b.y)/2-4); label.setAttribute('text-anchor','middle');
    label.setAttribute('font-size','9'); label.setAttribute('font-weight',selected?'700':'500'); label.setAttribute('fill',selected?'#f1d2a2':'#9da8bc'); label.textContent=e.branch; svg.appendChild(label);
  });

  nodes.forEach(n=>{
    const selected=pathSet.has(n.id); const g=document.createElementNS(NS,'g');
    const rect=document.createElementNS(NS,'rect');
    rect.setAttribute('x',n.x-nodeHalfW); rect.setAttribute('y',n.y-nodeHalfH); rect.setAttribute('width',nodeHalfW*2); rect.setAttribute('height',nodeHalfH*2); rect.setAttribute('rx','8');
    rect.setAttribute('fill',selected?'#2b1f20':'#111a2a'); rect.setAttribute('stroke',selected?'#d0a15f':'#556178'); rect.setAttribute('stroke-width',selected?'2.2':'1.05'); g.appendChild(rect);
    const t=document.createElementNS(NS,'text');
    t.setAttribute('x',n.x); t.setAttribute('y',n.y-6); t.setAttribute('text-anchor','middle'); t.setAttribute('font-size','8.3');
    t.setAttribute('font-weight',selected?'700':'600'); t.setAttribute('fill',selected?'#fff0d1':'#dfe6f5');
    const lines=n.is_leaf?[n.label,`P=${Math.round(n.positive_probability*100)}% · n=${n.samples}`]:[n.feature_label,`≤ ${n.threshold_label}? · n=${n.samples}`];
    lines.forEach((txt,i)=>{ const sp=document.createElementNS(NS,'tspan'); sp.setAttribute('x',n.x); sp.setAttribute('dy',i===0?'0':'12'); sp.textContent=txt; t.appendChild(sp); });
    g.appendChild(t); svg.appendChild(g);
  });
}

function escapeHtml(s){ return String(s??'').replace(/[&<>'"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch])); }
function escapeAttr(s){ return escapeHtml(s); }

init().catch(e=>showToast(e.message));
