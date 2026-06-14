/* ---------- deterministic RNG (for sampled point clouds) ---------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const rng=mulberry32(20260530);
function gauss(){let u,v;do{u=rng();}while(!u);do{v=rng();}while(!v);return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}

/* ---------- data ---------- */
const FEATURES=[
  {id:'mf_u',label:'MF',phon:'ʊ',cllr:0.420,sd:0.0650,eer:11.1,diagram:'ellipse',vowel:'ʊ'},
  {id:'mf_o',label:'MF',phon:'ɒ',cllr:0.762,sd:0.0458,eer:27.8,diagram:'ellipse',vowel:'ɒ'},
  {id:'ltfd',label:'LTFDs',phon:'',cllr:0.936,sd:0.0004,eer:37.6,diagram:'ltfd'},
  {id:'ltf0',label:'LTF0',phon:'',cllr:0.961,sd:0.0003,eer:42.0,diagram:'ltf0'},
  {id:'mfcc',label:'MFCCs',phon:'',cllr:0.923,sd:0.0918,eer:36.7,diagram:'mel'},
];
const VOWELS={
  'ʊ':{real:{m:[440,1020],c:[[1300,-600],[-600,7200]]}, fake:{m:[486,1128],c:[[620,-180],[-180,3400]]}},
  'ɒ':{real:{m:[600,955], c:[[1800,500],[500,5200]]},   fake:{m:[616,986], c:[[1500,420],[420,4600]]}},
};
const LTFD={real:[[505,95,1.0],[1490,150,0.82],[2520,180,0.58],[3500,210,0.34]],
            fake:[[540,78,1.05],[1545,120,0.74],[2430,150,0.5],[3380,170,0.26]]};
const LTF0={real:[[118,16,1.0]], fake:[[124,9,1.18]]};

/* ---------- geometry ---------- */
function ellipsePoints(mean,cov,scale){
  const a=cov[0][0],b=cov[0][1],c=cov[1][1];
  const tr=a+c, disc=Math.sqrt(Math.max(0,tr*tr/4-(a*c-b*b)));
  const rx=Math.sqrt(Math.max(tr/2+disc,0))*scale, ry=Math.sqrt(Math.max(tr/2-disc,0))*scale;
  const th=0.5*Math.atan2(2*b,a-c), ct=Math.cos(th), st=Math.sin(th);
  const pts=[];
  for(let i=0;i<=64;i++){
    const t=i/64*2*Math.PI, ex=rx*Math.cos(t), ey=ry*Math.sin(t);
    pts.push([mean[0]+ex*ct-ey*st, mean[1]+ex*st+ey*ct]);
  }
  return pts;
}
function sampleCloud(mean,cov,n){
  const a=cov[0][0],b=cov[0][1],c=cov[1][1];
  const l11=Math.sqrt(a), l21=b/l11, l22=Math.sqrt(Math.max(c-l21*l21,1e-6));
  const out=[];
  for(let i=0;i<n;i++){const z1=gauss(),z2=gauss();out.push([mean[0]+l11*z1, mean[1]+l21*z1+l22*z2]);}
  return out;
}

/* ---------- plotting helpers ---------- */
const W=560,H=420,M={t:24,r:22,b:54,l:62}, PW=W-M.l-M.r, PH=H-M.t-M.b;
const SVG_OPEN=`<svg class="plot" viewBox="0 0 ${W} ${H}">`;
const makeScale=(d0,d1,p0,p1)=>v=>p0+(v-d0)/(d1-d0)*(p1-p0);
const niceTicks=(min,max,n)=>Array.from({length:n+1},(_,i)=>min+(max-min)/n*i);

function axes(sx,sy,xr,yr,xlab,ylab,xfmt,yfmt,nx,ny){
  let g=`<rect x="${M.l}" y="${M.t}" width="${PW}" height="${PH}" fill="#fdfbff" stroke="none"/>`;
  niceTicks(xr[0],xr[1],nx).forEach(t=>{const x=sx(t);
    g+=`<line x1="${x}" y1="${M.t}" x2="${x}" y2="${M.t+PH}" stroke="var(--grid)" stroke-width="1"/>`;
    g+=`<text x="${x}" y="${M.t+PH+18}" fill="var(--ink-faint)" font-size="10.5" text-anchor="middle">${xfmt(t)}</text>`;});
  niceTicks(yr[0],yr[1],ny).forEach(t=>{const y=sy(t);
    g+=`<line x1="${M.l}" y1="${y}" x2="${M.l+PW}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    g+=`<text x="${M.l-9}" y="${y+3.5}" fill="var(--ink-faint)" font-size="10.5" text-anchor="end">${yfmt(t)}</text>`;});
  g+=`<rect x="${M.l}" y="${M.t}" width="${PW}" height="${PH}" fill="none" stroke="var(--ink-faint)" stroke-width="1"/>`;
  g+=`<text x="${M.l+PW/2}" y="${H-12}" fill="var(--ink-soft)" font-size="11.5" text-anchor="middle" font-weight="600">${xlab}</text>`;
  g+=`<text transform="translate(16,${M.t+PH/2}) rotate(-90)" fill="var(--ink-soft)" font-size="11.5" text-anchor="middle" font-weight="600">${ylab}</text>`;
  return g;
}

/* ---------- ellipse plot (F1 / F2) ---------- */
function renderEllipse(f){
  const v=VOWELS[f.vowel];
  const f1s=[v.real.m[0],v.fake.m[0]], f2s=[v.real.m[1],v.fake.m[1]];
  const f1min=Math.min(...f1s)-120, f1max=Math.max(...f1s)+120;
  const f2min=Math.min(...f2s)-260, f2max=Math.max(...f2s)+260;
  const sx=makeScale(f1min,f1max,M.l,M.l+PW);
  const sy=makeScale(f2min,f2max,M.t+PH,M.t); // invert
  const round=Math.round;
  let g=SVG_OPEN+axes(sx,sy,[f1min,f1max],[f2min,f2max],'F1 (Hz)','F2 (Hz)',round,round,5,5);
  const scale=2.4477; // 95% chi-square (2 dof)
  for(const k of ['real','fake']){
    const cfg=v[k], col=`var(--${k})`, fill=`var(--${k}-fill)`;
    sampleCloud(cfg.m,cfg.c,34).forEach(p=>{
      g+=`<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="2.4" fill="${col}" opacity="0.5"/>`;});
    const pts=ellipsePoints(cfg.m,cfg.c,scale).map(p=>`${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
    g+=`<polygon points="${pts}" fill="${fill}" stroke="${col}" stroke-width="2"/>`;
    g+=`<circle cx="${sx(cfg.m[0])}" cy="${sy(cfg.m[1])}" r="3.5" fill="#fff" stroke="${col}" stroke-width="2"/>`;
  }
  return g+`</svg>`;
}

/* ---------- density curves (LTFD / LTF0) ---------- */
const mixDensity=(peaks,x)=>peaks.reduce((y,[mu,sd,w])=>y+w*Math.exp(-0.5*((x-mu)/sd)**2),0);
function densityPlot(realP,fakeP,xr,xlab,nx){
  const sx=makeScale(xr[0],xr[1],M.l,M.l+PW), N=200, xs=[];
  let ymax=0;
  for(let i=0;i<=N;i++){const x=xr[0]+(xr[1]-xr[0])*i/N;xs.push(x);ymax=Math.max(ymax,mixDensity(realP,x),mixDensity(fakeP,x));}
  ymax*=1.12;
  const sy=makeScale(0,ymax,M.t+PH,M.t), base=sy(0);
  const path=peaks=>xs.map((x,i)=>(i?'L':'M')+sx(x).toFixed(1)+' '+sy(mixDensity(peaks,x)).toFixed(1)).join(' ');
  let g=SVG_OPEN+axes(sx,sy,xr,[0,ymax],xlab,'Relative density',Math.round,()=>'',nx,4);
  g+=`<path d="${path(realP)} L ${sx(xr[1])} ${base} L ${sx(xr[0])} ${base} Z" fill="var(--real-fill)" stroke="none"/>`;
  g+=`<path d="${path(realP)}" fill="none" stroke="var(--real)" stroke-width="2.4"/>`;
  g+=`<path d="${path(fakeP)}" fill="none" stroke="var(--fake)" stroke-width="2.4" stroke-dasharray="6,4"/>`;
  return g+`</svg>`;
}

/* ---------- z-scored log-Mel filterbank energy ---------- */
function melData(){
  const NB=40,real=[],fake=[],ciR=[],ciF=[];
  for(let k=0;k<NB;k++){
    const t=k/(NB-1);
    const base=1.15*Math.exp(-(((t-0.18)/0.16)**2)) - 0.9*t + 0.15*Math.sin(t*7);
    const dev=-0.95*Math.max(0,t-0.55)*1.8 + 0.18*Math.exp(-(((t-0.42)/0.08)**2));
    real.push(base); fake.push(base+dev); ciR.push(0.18+0.10*t); ciF.push(0.22+0.16*t);
  }
  return {NB,real,fake,ciR,ciF};
}
function renderMel(){
  const d=melData(), sx=makeScale(1,d.NB,M.l,M.l+PW);
  const allv=[];
  for(let k=0;k<d.NB;k++)allv.push(d.real[k]+d.ciR[k],d.real[k]-d.ciR[k],d.fake[k]+d.ciF[k],d.fake[k]-d.ciF[k]);
  const ymin=Math.min(...allv)-0.15, ymax=Math.max(...allv)+0.15;
  const sy=makeScale(ymin,ymax,M.t+PH,M.t);
  const X=k=>sx(k+1).toFixed(1);
  let g=SVG_OPEN+axes(sx,sy,[1,d.NB],[ymin,ymax],'Mel filterbank channel (low to high freq)','Energy (z-score)',Math.round,v=>v.toFixed(1),8,5);
  g+=`<line x1="${M.l}" y1="${sy(0)}" x2="${M.l+PW}" y2="${sy(0)}" stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="2,3"/>`;
  const band=(arr,ci,fill)=>{
    let up='',dn='';
    for(let k=0;k<d.NB;k++)up+=(k?'L':'M')+X(k)+' '+sy(arr[k]+ci[k]).toFixed(1)+' ';
    for(let k=d.NB-1;k>=0;k--)dn+='L'+X(k)+' '+sy(arr[k]-ci[k]).toFixed(1)+' ';
    g+=`<path d="${up}${dn}Z" fill="${fill}" stroke="none"/>`;
  };
  const line=(arr,col,dash)=>{
    let p='';
    for(let k=0;k<d.NB;k++)p+=(k?'L':'M')+X(k)+' '+sy(arr[k]).toFixed(1)+' ';
    g+=`<path d="${p}" fill="none" stroke="${col}" stroke-width="2.4" stroke-dasharray="${dash}"/>`;
  };
  band(d.real,d.ciR,'var(--real-fill)');
  band(d.fake,d.ciF,'var(--fake-fill)');
  line(d.real,'var(--real)','none');
  line(d.fake,'var(--fake)','6,4');
  return g+`</svg>`;
}

/* ---------- per-diagram render + copy ---------- */
const RENDER={
  ellipse:renderEllipse,
  ltfd:()=>densityPlot(LTFD.real,LTFD.fake,[0,4000],'Frequency (Hz): F3',8),
  ltf0:()=>densityPlot(LTF0.real,LTF0.fake,[60,260],'Fundamental frequency F0 (Hz)',5),
  mel:renderMel,
};
const TITLES={ellipse:'F1 × F2 confidence ellipses',ltfd:'Long-term formant distribution',ltf0:'Long-term F0 distribution',mel:'z-scored log-Mel filterbank energy'};
const INTERP={
  ellipse:f=>`<h3>Explanation</h3>
    <p>Each point is one measured token of the vowel <span class="phon">[${f.vowel}]</span> at its formant midpoint; each shaded shell is the 95% confidence ellipse derived from the <strong>principal axes of that token's covariance matrix</strong> (F1 × F2).</p>`,
  ltfd:`<h3>Explanation</h3>
    <p>The long-term formant distribution pools every voiced frame in the recording into a density over frequency, with peaks near F1-F4. It describes a speaker's overall resonance profile rather than a single sound.</p>`,
  ltf0:`<h3>Explanation</h3>
    <p>The long-term F0 distribution is the density of fundamental frequency (perceived pitch) across the whole utterance. Its width reflects intonation range.</p>`,
  mel:`<h3>Explanation</h3>
    <p>Mean z-scored log-Mel filterbank energy per channel (low to high frequency); shaded regions are 95% confidence intervals across utterances.</p>`,
};

/* ---------- table + interaction ---------- */
const cllrColor=c=>c<0.5?'var(--good)':c<0.85?'var(--mid)':'var(--weak)';
let current=FEATURES[0];

FEATURES.forEach(f=>{
  const tr=document.createElement('tr');
  tr.dataset.id=f.id;
  const name=f.phon?`${f.label} <span class="vowel phon">${f.phon}</span>`:f.label;
  tr.innerHTML=`<td>${name}</td>
    <td>${f.cllr.toFixed(3)}<span class="meter"><i style="width:${Math.min(100,f.cllr*100)}%;background:${cllrColor(f.cllr)}"></i></span></td>
    <td>${f.sd.toFixed(4)}</td>
    <td>${f.eer.toFixed(1)}</td>`;
  tr.addEventListener('mouseenter',()=>select(f));
  tr.addEventListener('click',()=>select(f));
  document.querySelector('#resultsTable tbody').appendChild(tr);
});

function render(){
  const f=current;
  document.querySelectorAll('#resultsTable tbody tr').forEach(tr=>tr.classList.toggle('active',tr.dataset.id===f.id));
  const sub=(f.diagram==='ellipse'?`Vowel [${f.vowel}]`:f.label)+`: C\u2097\u2097\u1d63 ${f.cllr.toFixed(3)}: EER ${f.eer.toFixed(1)}%`;
  document.getElementById('viewerTitle').innerHTML=`<h2>${TITLES[f.diagram]}</h2><small>${sub}</small>`;
  document.getElementById('plot').innerHTML=RENDER[f.diagram](f);
  const interp=INTERP[f.diagram];
  document.getElementById('interp').innerHTML=typeof interp==='function'?interp(f):interp;
}
function select(f){current=f;render();}

render();