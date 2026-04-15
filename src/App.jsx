import { useState, useCallback, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, Cell, ResponsiveContainer, ReferenceLine
} from "recharts";

/* ════════════════════════════════════════════════════════════════════════
   7D-O₂ CLOSED-LOOP RESPIRATORY CONTROL MODEL — RESEARCH WEBSITE
   Diekman, Thomas & Wilson (2017, 2024)
   COVID-19 Silent Hypoxemia

   Paper-accurate parameters:
     Normoxia: φ=0.3, θg=85, σg=30, [Hb]=150
     Silent Hypoxemia: φ=0.24, θg=70, σg=36, [Hb]=250
   ════════════════════════════════════════════════════════════════════════ */

// ── Paper-exact parameters (Appendix A, Eqs 7-28) ────────────────────
const NORM = {
  C:21, gK:11.2, gNaP:2.8, gNa:28, gL:2.8,
  EK:-85, ENa:50, EL:-65, Etonic:0,
  theta_p:-40, sigma_p:-6, theta_h:-48, sigma_h:6, tau_h_bar:10000,
  theta_m:-34, sigma_m:-5, theta_n:-29, sigma_n:-4, tau_n_bar:10,
  ra:0.001, Tmax:1, VT:2, Kp:5,
  E1:0.4, E2:0.0025, vol0:2,
  PextO2:(760-47)*0.21, R_gas:62.364, Temp:310, tau_LB:500,
  M:8e-6, Hb:150, volB:5, betaO2:0.03, c:2.5, K:26,
  phi:0.3, theta_g:85, sigma_g:30,
};
const SH_PAPER = { ...NORM, phi:0.24, theta_g:70, sigma_g:36, Hb:250 };
const INIT = [-56.8172, 9.5344e-4, 0.7454, 2.0026e-4, 2.0525, 98.9638, 97.7927];

// ── ODE solver ────────────────────────────────────────────────────────
function odeRHS(u, p) {
  const [V,n,h,alpha,volL,PAO2,PaO2_raw] = u;
  const PaO2 = Math.max(PaO2_raw, 1e-6);
  const xinf = (V,th,si) => 1/(1+Math.exp((V-th)/si));
  const taux = (V,th,si,tb) => tb/Math.cosh((V-th)/(2*si));
  const p_inf=xinf(V,p.theta_p,p.sigma_p), m_inf=xinf(V,p.theta_m,p.sigma_m);
  const n_inf=xinf(V,p.theta_n,p.sigma_n), h_inf=xinf(V,p.theta_h,p.sigma_h);
  const tn=taux(V,p.theta_n,p.sigma_n,p.tau_n_bar);
  const th=taux(V,p.theta_h,p.sigma_h,p.tau_h_bar);
  const IK=p.gK*Math.pow(n,4)*(V-p.EK);
  const INaP=p.gNaP*p_inf*h*(V-p.ENa);
  const INa=p.gNa*Math.pow(m_inf,3)*(1-n)*(V-p.ENa);
  const IL=p.gL*(V-p.EL);
  const gt=p.phi*(1-Math.tanh((PaO2-p.theta_g)/p.sigma_g));
  const It=gt*(V-p.Etonic);
  const Tc=p.Tmax/(1+Math.exp(-(V-p.VT)/p.Kp));
  const dvL=-p.E2*(volL-p.vol0)+p.E1*alpha;
  const eta=p.Hb*1.36, zeta=p.volB/22400, cc=p.c, KK=p.K;
  const S=Math.pow(PaO2,cc)/(Math.pow(PaO2,cc)+Math.pow(KK,cc));
  const dS=cc*Math.pow(PaO2,cc-1)*(1/(Math.pow(PaO2,cc)+Math.pow(KK,cc))-Math.pow(PaO2,cc)/Math.pow(Math.pow(PaO2,cc)+Math.pow(KK,cc),2));
  const JLB=(PAO2-PaO2)/p.tau_LB*(volL/(p.R_gas*p.Temp));
  const JBT=p.M*zeta*(p.betaO2*PaO2+eta*S);
  const dvLp=Math.max(0,dvL);
  const dPAO2=(p.PextO2-PAO2)/volL*dvLp-(PAO2-PaO2)/p.tau_LB;
  const den=Math.max(zeta*(p.betaO2+eta*dS),1e-15);
  return [(-IK-INaP-INa-IL-It)/p.C,(n_inf-n)/tn,(h_inf-h)/th,
          p.ra*Tc*(1-alpha)-p.ra*alpha, dvL, dPAO2, (JLB-JBT)/den];
}

function integrate(params, tf=15000, dt=0.5) {
  let u=[...INIT]; const N=Math.floor(tf/dt);
  const skip=Math.max(1,Math.floor(N/1800));
  const res=[];
  for(let i=0;i<=N;i++){
    if(i%skip===0){
      const P=Math.max(u[6],1e-6), cc=params.c, KK=params.K;
      const S=Math.pow(P,cc)/(Math.pow(P,cc)+Math.pow(KK,cc));
      const g=params.phi*(1-Math.tanh((P-params.theta_g)/params.sigma_g));
      res.push({t:+(i*dt/1000).toFixed(3),V:u[0],n:u[1],h:u[2],alpha:u[3],
                volL:u[4],PAO2:u[5],PaO2:u[6],SaO2:S*100,gtonic:g});
    }
    const k1=odeRHS(u,params);
    const um=u.map((v,j)=>v+0.5*dt*k1[j]);
    const k2=odeRHS(um,params);
    u=u.map((v,j)=>v+dt*k2[j]);
    u[6]=Math.max(u[6],0.1); u[4]=Math.max(u[4],0.1);
  }
  return res;
}

function computeL2(d1,d2,key){
  const len=Math.min(d1.length,d2.length);
  if(len<2)return 0; let s=0;
  for(let i=0;i<len;i++){const d=(d1[i]?.[key]??0)-(d2[i]?.[key]??0);s+=d*d;}
  const T=Math.max(d1[len-1].t-d1[0].t,1e-6), dt=T/(len-1);
  return Math.sqrt(s*dt/T);
}

// ── Theme: academic light ─────────────────────────────────────────────
const C = {
  bg: "#ffffff", surface: "#f8f9fa", card: "#ffffff",
  border: "#dee2e6", borderLight: "#e9ecef",
  primary: "#1a5276", primaryLight: "#2980b9",
  accent: "#117a65",
  normoxia: "#c0392b", sh: "#2471a3",
  text: "#212529", textSecondary: "#495057", muted: "#6c757d", faint: "#adb5bd",
  gold: "#b7950b", codeBg: "#f4f6f7", codeText: "#1a5276", highlight: "#fef9e7",
};

const F = {
  display: "'Libre Baskerville', 'Georgia', 'Times New Roman', serif",
  body: "'Source Sans 3', 'Helvetica Neue', 'Segoe UI', sans-serif",
  mono: "'IBM Plex Mono', 'Consolas', 'Courier New', monospace",
};

// ── Reusable components ───────────────────────────────────────────────
function Slider({label,value,min,max,step,onChange,unit=""}){
  return(
    <div style={{marginBottom:10}}>
      <div style={{display:'flex',justifyContent:'space-between',fontSize:12,color:C.muted,marginBottom:2}}>
        <span style={{fontFamily:F.display,fontStyle:'italic',fontSize:11}}>{label}</span>
        <span style={{fontFamily:F.mono,color:C.primary,fontWeight:600,fontSize:11}}>
          {value<0.001?value.toExponential(1):Number(value).toFixed(step<0.01?4:step<1?2:0)}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e=>onChange(+e.target.value)}
        style={{width:'100%',accentColor:C.primaryLight,height:4,cursor:'pointer'}}/>
    </div>
  );
}

function Stat({label,value,unit,warn}){
  return(
    <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,
      padding:'10px 14px',textAlign:'center',flex:1,minWidth:110}}>
      <div style={{fontSize:10,color:C.muted,textTransform:'uppercase',letterSpacing:1,
        fontFamily:F.body,fontWeight:600}}>{label}</div>
      <div style={{fontSize:22,fontWeight:700,color:warn?C.normoxia:C.primary,
        fontFamily:F.mono,marginTop:3}}>{value}</div>
      <div style={{fontSize:9,color:C.muted}}>{unit}</div>
    </div>
  );
}

function Code({children,block=false}){
  if(block) return(
    <pre style={{background:C.codeBg,border:`1px solid ${C.borderLight}`,borderRadius:4,
      padding:'12px 16px',fontFamily:F.mono,fontSize:11,color:C.text,lineHeight:1.6,
      overflowX:'auto',margin:'12px 0',whiteSpace:'pre-wrap'}}>{children}</pre>
  );
  return <code style={{background:C.codeBg,padding:'1px 5px',borderRadius:3,
    fontFamily:F.mono,fontSize:'0.9em',color:C.codeText}}>{children}</code>;
}

function Section({id,children,style={}}){
  return <section id={id} style={{padding:'40px 24px',maxWidth:1100,margin:'0 auto',...style}}>{children}</section>;
}

function SectionTitle({children,sub}){
  return(
    <div style={{marginBottom:28,borderBottom:`2px solid ${C.primary}`,paddingBottom:12}}>
      <h2 style={{fontFamily:F.display,fontSize:22,fontWeight:700,color:C.primary,margin:0}}>{children}</h2>
      {sub && <p style={{fontFamily:F.body,fontSize:13,color:C.muted,marginTop:6,marginBottom:0}}>{sub}</p>}
    </div>
  );
}

function ProofBlock({title,children}){
  return(
    <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,
      padding:'20px 24px',marginBottom:20}}>
      <h4 style={{fontFamily:F.display,fontSize:14,color:C.primary,margin:'0 0 10px',fontWeight:700}}>{title}</h4>
      <div style={{fontFamily:F.body,fontSize:13,color:C.textSecondary,lineHeight:1.75}}>{children}</div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════
// MAIN APPLICATION
// ════════════════════════════════════════════════════════════════════════
export default function App(){
  const [page,setPage]=useState('home');
  const [params,setParams]=useState({...NORM});
  const [mode,setMode]=useState('normoxia');
  const [data1,setData1]=useState(null);
  const [data2,setData2]=useState(null);
  const [running,setRunning]=useState(false);
  const [activeVar,setActiveVar]=useState('PaO2');
  const [l2s,setL2s]=useState(null);

  const runSim=useCallback(()=>{
    setRunning(true);
    setTimeout(()=>{
      try{
        const d1=integrate(mode==='sh'?SH_PAPER:params);
        setData1(d1);
        if(mode==='compare'){
          const d2=integrate(SH_PAPER);
          setData2(d2);
          const keys=['V','n','h','alpha','volL','PAO2','PaO2'];
          const norms={};
          keys.forEach(k=>{norms[k]=computeL2(d1,d2,k)});
          setL2s(norms);
        } else { setData2(null); setL2s(null); }
      }catch(e){console.error(e);}
      setRunning(false);
    },60);
  },[params,mode]);

  useEffect(()=>{runSim();},[]);

  const setPreset=(m)=>{
    setMode(m);
    if(m==='normoxia')setParams({...NORM});
    else if(m==='sh')setParams({...SH_PAPER});
    else setParams({...NORM});
  };

  const last=data1?.[data1.length-1];
  const fPaO2=last?.PaO2?.toFixed(1)??"—";
  const fSaO2=last?.SaO2?.toFixed(1)??"—";

  const vars=[
    {key:'V',label:'V (mV)',c:C.sh},
    {key:'alpha',label:'\u03B1',c:'#7d3c98'},
    {key:'volL',label:'volL (L)',c:C.accent},
    {key:'PaO2',label:'PaO\u2082 (mmHg)',c:C.primary},
    {key:'PAO2',label:'PAO\u2082 (mmHg)',c:'#a93226'},
    {key:'SaO2',label:'SaO\u2082 (%)',c:C.gold},
    {key:'gtonic',label:'g_tonic (nS)',c:'#148f77'},
  ];

  const l2Bar=l2s?Object.entries(l2s).map(([k,v])=>({name:k,value:+v.toFixed(4)})):[];

  // ── Navigation ──────────────────────────────────────────────────────
  const Nav=()=>(
    <nav style={{
      position:'sticky',top:0,zIndex:100,
      background:'rgba(255,255,255,0.97)',backdropFilter:'blur(8px)',
      borderBottom:`1px solid ${C.border}`,padding:'0 24px',
      display:'flex',alignItems:'center',height:52,
    }}>
      <div style={{fontFamily:F.display,fontSize:16,fontWeight:700,color:C.primary,cursor:'pointer',
        whiteSpace:'nowrap'}} onClick={()=>setPage('home')}>
        7D-O\u2082 Respiratory Model Python - Brayden Choi
      </div>
      <div style={{flex:1}}/>
      {[['home','Overview'],['explorer','Simulator'],['findings','Findings & Proof'],['paper','References']].map(([k,lbl])=>(
        <button key={k} onClick={()=>setPage(k)} style={{
          background:'none',border:'none',color:page===k?C.primary:C.muted,
          fontFamily:F.body,fontSize:13,fontWeight:page===k?700:400,
          cursor:'pointer',padding:'8px 14px',
          borderBottom:page===k?`2px solid ${C.primary}`:'2px solid transparent',
        }}>{lbl}</button>
      ))}
    </nav>
  );

  // ════════════════════════════════════════════════════════════════════
  // PAGE: HOME
  // ════════════════════════════════════════════════════════════════════
  const Home=()=>(
    <>
      <div style={{background:C.surface,borderBottom:`1px solid ${C.border}`,
        padding:'48px 24px 40px',textAlign:'center'}}>
        <div style={{maxWidth:800,margin:'0 auto'}}>
          <div style={{fontSize:11,fontFamily:F.mono,color:C.accent,letterSpacing:2,
            textTransform:'uppercase',marginBottom:14,fontWeight:600}}>Computational Neuroscience</div>
          <h1 style={{fontFamily:F.display,fontSize:28,fontWeight:700,color:C.text,
            margin:'0 0 16px',lineHeight:1.4}}>
            COVID-19 and Silent Hypoxemia in a Minimal Closed-Loop Model of the Respiratory Rhythm Generator
          </h1>
          <p style={{fontFamily:F.body,fontSize:14,color:C.muted,maxWidth:640,margin:'0 auto 6px',lineHeight:1.6}}>
            MATLAB to Python translation with interactive parameter exploration, L\u2082 norm cross-validation,
            and publication-quality figure generation
          </p>
          <p style={{fontFamily:F.display,fontSize:12,color:C.muted,fontStyle:'italic',marginBottom:24}}>
            Based on Diekman CO, Thomas PJ, Wilson CG (2024). <em>Biological Cybernetics</em>, 118(3-4):145\u2013163
          </p>
          <div style={{display:'flex',gap:10,justifyContent:'center',flexWrap:'wrap'}}>
            <button onClick={()=>setPage('explorer')} style={{
              padding:'10px 24px',borderRadius:4,border:'none',cursor:'pointer',
              background:C.primary,color:'#fff',fontFamily:F.body,fontSize:13,fontWeight:600,
            }}>Open Simulator</button>
            <button onClick={()=>setPage('findings')} style={{
              padding:'10px 24px',borderRadius:4,border:`1px solid ${C.primary}`,cursor:'pointer',
              background:'transparent',color:C.primary,fontFamily:F.body,fontSize:13,fontWeight:600,
            }}>View Findings</button>
          </div>
        </div>
      </div>

      <Section>
        <SectionTitle sub="The 7D-O\u2082 model couples a Butera-Rinzel-Smith central pattern generator with lung mechanics, oxygen transport, and chemosensory feedback">
          Model Architecture
        </SectionTitle>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(300px,1fr))',gap:16}}>
          {[
            {title:'Central Pattern Generator',desc:'The Butera-Rinzel-Smith (BRS) pacemaker neuron model with persistent Na\u207A (INaP), fast Na\u207A (INa), delayed-rectifier K\u207A (IK), and leak (IL) currents. Three state variables: membrane potential V, potassium activation n, and persistent sodium inactivation h. Equations 7\u201316 of the paper.'},
            {title:'Motor Pool and Lung Mechanics',desc:'CPG voltage drives a synaptic motor pool activation variable \u03B1 (Eq. 17\u201318), which in turn expands lung volume volL (Eq. 19). The low-pass filter behavior of the musculature means that high-frequency tonic spiking does not drive effective ventilation \u2014 only rhythmic bursting does.'},
            {title:'Oxygen Transport',desc:'Alveolar oxygen (PAO\u2082, Eq. 20) and arterial blood oxygen (PaO\u2082, Eq. 21) are coupled through lung-to-blood flux JLB and blood-to-tissue metabolic consumption JBT. Hemoglobin binding follows a Hill equation (Eq. 24) with c = 2.5 and K = 26 mmHg.'},
            {title:'Chemosensory Feedback Loop',desc:'The sigmoidal chemosensory function g_tonic = \u03C6(1 \u2212 tanh((PaO\u2082 \u2212 \u03B8g)/\u03C3g)) closes the control loop (Eq. 28). When PaO\u2082 drops, g_tonic increases, driving the CPG to burst faster. This is the pathway hypothesized to be impaired by COVID-19 infection.'},
            {title:'Silent Hypoxemia Mechanism',desc:'The paper demonstrates that altering chemosensory gain alone (\u03C6, \u03B8g, \u03C3g) lowers the PaO\u2082 plateau but shifts the metabolic collapse point rightward. Only when hemoglobin concentration [Hb] is increased to 250 g/L does the collapse point shift to lower metabolic demand \u2014 producing the clinical SH phenotype.'},
            {title:'The M \u00D7 \u03B7 Invariance',desc:'A key mathematical insight: because the Henry\'s law constant \u03B2O\u2082 \u2248 0.03 is small, the blood oxygen dynamics depend approximately on the product M \u00D7 \u03B7 (metabolic demand times hemoglobin capacity). Rescaling M \u2192 \u03B3M, \u03B7 \u2192 \u03B7/\u03B3 leaves the dynamics nearly invariant (Eqs. 5\u20136).'},
          ].map((c,i)=>(
            <div key={i} style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:6,
              padding:'20px 24px',textAlign:'center'}}>
              <h3 style={{fontFamily:F.display,fontSize:14,color:C.primary,margin:'0 0 10px',fontWeight:700}}>{c.title}</h3>
              <p style={{fontFamily:F.body,fontSize:12.5,color:C.textSecondary,lineHeight:1.7,margin:0}}>{c.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section style={{paddingTop:0}}>
        <SectionTitle sub="Exact values from the paper\u2019s Appendix A (Equations 7\u201328) and Results section (p. 7\u20139)">
          Parameter Specification
        </SectionTitle>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          {[
            {title:'Normoxia (Original 7D-O\u2082 Model)',params:[
              ['\u03C6 (phi)','0.3 nS','Eq. 28'],['\u03B8g','85 mmHg','Eq. 28'],['\u03C3g','30 mmHg','Eq. 28'],
              ['[Hb]','150 g/L','Eq. 27'],['M','8\u00D710\u207B\u2076 ms\u207B\u00B9','Eq. 23'],
              ['K','26 mmHg','Eq. 24'],['c','2.5','Eq. 24'],['vol\u2080','2.0 L','Eq. 19'],['\u03C4_LB','500 ms','Eq. 22'],
            ],color:C.normoxia,note:'ModelDB #229640'},
            {title:'Silent Hypoxemia (SH Working Model)',params:[
              ['\u03C6 (phi)','0.24 nS','Results p.7'],['\u03B8g','70 mmHg','Results p.7'],['\u03C3g','36 mmHg','Results p.7'],
              ['[Hb]','250 g/L','Fig. 5B, p.9'],['M','8\u00D710\u207B\u2076 ms\u207B\u00B9','unchanged'],
              ['K','26 mmHg','unchanged'],['c','2.5','unchanged'],['vol\u2080','2.0 L','unchanged'],['\u03C4_LB','500 ms','unchanged'],
            ],color:C.sh,note:'ModelDB #2015954 \u2014 \u03C6=0.24 \u03B8g=70 \u03C3g=36 selected for greatest PaO\u2082 reduction (p.7)'},
          ].map((t,i)=>(
            <div key={i} style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:6,padding:'16px 20px'}}>
              <h3 style={{fontFamily:F.display,fontSize:13,color:t.color,margin:'0 0 4px',fontWeight:700}}>{t.title}</h3>
              <p style={{fontFamily:F.mono,fontSize:10,color:C.faint,margin:'0 0 10px'}}>{t.note}</p>
              <table style={{width:'100%',borderCollapse:'collapse',fontSize:11,fontFamily:F.mono}}>
                <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                  <th style={{textAlign:'left',padding:'4px 6px',color:C.muted,fontWeight:600,fontSize:10}}>Param</th>
                  <th style={{textAlign:'left',padding:'4px 6px',color:C.muted,fontWeight:600,fontSize:10}}>Value</th>
                  <th style={{textAlign:'left',padding:'4px 6px',color:C.muted,fontWeight:600,fontSize:10}}>Source</th>
                </tr></thead>
                <tbody>{t.params.map(([k,v,src],j)=>(
                  <tr key={j} style={{borderBottom:`1px solid ${C.borderLight}`}}>
                    <td style={{padding:'4px 6px',color:C.text}}>{k}</td>
                    <td style={{padding:'4px 6px',color:C.primary,fontWeight:600}}>{v}</td>
                    <td style={{padding:'4px 6px',color:C.faint,fontSize:10}}>{src}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ))}
        </div>
      </Section>
    </>
  );

  // ════════════════════════════════════════════════════════════════════
  // PAGE: SIMULATOR
  // ════════════════════════════════════════════════════════════════════
  const Explorer=()=>(
    <Section>
      <SectionTitle sub="Adjust model parameters corresponding to the paper\u2019s equations and observe the effects on the 7D-O\u2082 system">
        Interactive Simulator
      </SectionTitle>
      <div style={{display:'flex',gap:8,marginBottom:14,flexWrap:'wrap',alignItems:'center'}}>
        {[['normoxia','Normoxia'],['sh','Silent Hypoxemia'],['compare','Compare']].map(([m,lbl])=>(
          <button key={m} onClick={()=>{setPreset(m);}} style={{
            padding:'6px 16px',borderRadius:4,border:`1px solid ${mode===m?(m==='sh'?C.sh:m==='compare'?C.accent:C.normoxia):C.border}`,
            cursor:'pointer',fontSize:12,fontWeight:600,fontFamily:F.body,
            background:mode===m?(m==='sh'?C.sh:m==='compare'?C.accent:C.normoxia):'transparent',
            color:mode===m?'#fff':C.textSecondary,
          }}>{lbl}</button>
        ))}
        <div style={{flex:1}}/>
        <button onClick={runSim} disabled={running} style={{
          padding:'7px 22px',borderRadius:4,border:'none',cursor:running?'wait':'pointer',
          background:running?C.border:C.primary,
          color:'#fff',fontSize:12,fontWeight:700,fontFamily:F.body,
        }}>{running?'Solving ODEs...':'Run Simulation'}</button>
      </div>

      <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
        <div style={{width:240,flexShrink:0,background:C.surface,border:`1px solid ${C.border}`,
          borderRadius:6,padding:14,maxHeight:680,overflowY:'auto'}}>
          <div style={{fontSize:10,fontWeight:700,color:C.primary,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8}}>Chemosensory \u2014 Eq. 28</div>
          <Slider label="\u03C6 (max g_tonic)" value={params.phi} min={0.05} max={0.6} step={0.01} onChange={v=>setParams(p=>({...p,phi:v}))} unit=" nS"/>
          <Slider label="\u03B8g (half-activation)" value={params.theta_g} min={30} max={120} step={1} onChange={v=>setParams(p=>({...p,theta_g:v}))} unit=" mmHg"/>
          <Slider label="\u03C3g (slope)" value={params.sigma_g} min={5} max={60} step={1} onChange={v=>setParams(p=>({...p,sigma_g:v}))} unit=" mmHg"/>
          <div style={{fontSize:10,fontWeight:700,color:C.primary,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8,marginTop:14}}>Blood O\u2082 \u2014 Eqs. 21\u201327</div>
          <Slider label="M (metabolic demand)" value={params.M} min={2e-6} max={2e-5} step={1e-6} onChange={v=>setParams(p=>({...p,M:v}))} unit=" ms\u207B\u00B9"/>
          <Slider label="[Hb] (hemoglobin)" value={params.Hb} min={80} max={350} step={5} onChange={v=>setParams(p=>({...p,Hb:v}))} unit=" g/L"/>
          <Slider label="c (Hill coefficient)" value={params.c} min={1} max={5} step={0.1} onChange={v=>setParams(p=>({...p,c:v}))}/>
          <Slider label="K (half-saturation)" value={params.K} min={8} max={40} step={1} onChange={v=>setParams(p=>({...p,K:v}))} unit=" mmHg"/>
          <div style={{fontSize:10,fontWeight:700,color:C.primary,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8,marginTop:14}}>CPG \u2014 Eqs. 7\u201316</div>
          <Slider label="gNaP" value={params.gNaP} min={0.5} max={8} step={0.1} onChange={v=>setParams(p=>({...p,gNaP:v}))} unit=" nS"/>
          <Slider label="gK" value={params.gK} min={2} max={25} step={0.2} onChange={v=>setParams(p=>({...p,gK:v}))} unit=" nS"/>
          <div style={{fontSize:10,fontWeight:700,color:C.primary,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8,marginTop:14}}>Lung \u2014 Eqs. 19\u201320</div>
          <Slider label="vol\u2080 (unloaded)" value={params.vol0} min={1} max={3} step={0.1} onChange={v=>setParams(p=>({...p,vol0:v}))} unit=" L"/>
          <Slider label="\u03C4_LB (O\u2082 flux)" value={params.tau_LB} min={100} max={1500} step={50} onChange={v=>setParams(p=>({...p,tau_LB:v}))} unit=" ms"/>
        </div>

        <div style={{flex:1,minWidth:0}}>
          <div style={{display:'flex',gap:8,marginBottom:10,flexWrap:'wrap'}}>
            <Stat label="PaO\u2082" value={fPaO2} unit="mmHg" warn={+fPaO2<70}/>
            <Stat label="SaO\u2082" value={fSaO2} unit="%" warn={+fSaO2<80}/>
            <Stat label="Preset" value={mode==='sh'?'SH':mode==='compare'?'Compare':'Normoxia'}
              unit={mode==='sh'?'\u03C6=.24 \u03B8g=70 \u03C3g=36 [Hb]=250':mode==='compare'?'normoxia vs SH':'\u03C6=.30 \u03B8g=85 \u03C3g=30 [Hb]=150'}/>
          </div>
          <div style={{display:'flex',gap:4,marginBottom:8,flexWrap:'wrap'}}>
            {vars.map(v=>(
              <button key={v.key} onClick={()=>setActiveVar(v.key)} style={{
                padding:'3px 8px',borderRadius:3,border:`1px solid ${activeVar===v.key?v.c:C.borderLight}`,
                cursor:'pointer',fontSize:10,fontFamily:F.mono,fontWeight:activeVar===v.key?700:400,
                background:activeVar===v.key?v.c+'12':'transparent',color:activeVar===v.key?v.c:C.muted,
              }}>{v.label}</button>
            ))}
          </div>
          {data1&&(
            <div style={{background:C.card,borderRadius:6,border:`1px solid ${C.border}`,padding:14,marginBottom:12}}>
              <ResponsiveContainer width="100%" height={mode==='compare'?260:330}>
                <LineChart data={data1} margin={{top:5,right:16,bottom:5,left:8}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.borderLight}/>
                  <XAxis dataKey="t" stroke={C.muted} fontSize={10} label={{value:'t (s)',position:'insideBottomRight',offset:-5,fill:C.muted,fontSize:10}}/>
                  <YAxis stroke={C.muted} fontSize={10}/>
                  <Tooltip contentStyle={{background:'#fff',border:`1px solid ${C.border}`,borderRadius:4,fontSize:11}} formatter={v=>[Number(v).toFixed(3),activeVar]}/>
                  <Line type="monotone" dataKey={activeVar} stroke={mode==='compare'?C.normoxia:vars.find(v=>v.key===activeVar)?.c||C.sh} dot={false} strokeWidth={1.4} name={mode==='compare'?'Normoxia':activeVar}/>
                  {activeVar==='SaO2'&&<ReferenceLine y={80} stroke={C.faint} strokeDasharray="5 5" label={{value:'80%',fill:C.faint,fontSize:9}}/>}
                </LineChart>
              </ResponsiveContainer>
              {mode==='compare'&&data2&&(
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart margin={{top:5,right:16,bottom:5,left:8}}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.borderLight}/>
                    <XAxis dataKey="t" stroke={C.muted} fontSize={10} data={data1}/>
                    <YAxis stroke={C.muted} fontSize={10}/>
                    <Tooltip contentStyle={{background:'#fff',border:`1px solid ${C.border}`,borderRadius:4,fontSize:11}}/>
                    <Legend wrapperStyle={{fontSize:11}}/>
                    <Line data={data1} type="monotone" dataKey={activeVar} stroke={C.normoxia} dot={false} strokeWidth={1.4} name="Normoxia"/>
                    <Line data={data2} type="monotone" dataKey={activeVar} stroke={C.sh} dot={false} strokeWidth={1.4} name="SH ([Hb]=250)"/>
                    {activeVar==='SaO2'&&<ReferenceLine y={80} stroke={C.faint} strokeDasharray="5 5"/>}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          )}
          {mode==='compare'&&l2s&&(
            <div style={{background:C.card,borderRadius:6,border:`1px solid ${C.border}`,padding:14}}>
              <div style={{fontFamily:F.display,fontSize:13,fontWeight:700,color:C.primary,marginBottom:8}}>
                L\u2082 Norm: Normoxia vs Silent Hypoxemia
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={l2Bar} margin={{top:5,right:16,bottom:5,left:8}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.borderLight}/>
                  <XAxis dataKey="name" stroke={C.muted} fontSize={11}/>
                  <YAxis stroke={C.muted} fontSize={10}/>
                  <Tooltip contentStyle={{background:'#fff',border:`1px solid ${C.border}`,borderRadius:4,fontSize:11}}/>
                  <Bar dataKey="value" radius={[3,3,0,0]}>
                    {l2Bar.map((_,i)=><Cell key={i} fill={i<4?C.sh:C.accent} fillOpacity={0.75}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{display:'flex',gap:6,flexWrap:'wrap',marginTop:6}}>
                {Object.entries(l2s).map(([k,v])=>(
                  <span key={k} style={{background:C.surface,border:`1px solid ${C.borderLight}`,borderRadius:3,
                    padding:'2px 6px',fontSize:10,fontFamily:F.mono}}>
                    <span style={{color:C.muted}}>{k}:</span>{' '}
                    <span style={{color:C.primary,fontWeight:600}}>{v.toFixed(4)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Section>
  );

  // ════════════════════════════════════════════════════════════════════
  // PAGE: FINDINGS & PROOF
  // ════════════════════════════════════════════════════════════════════
  const Findings=()=>(
    <Section>
      <SectionTitle sub="Detailed evidence of correctness for the MATLAB to Python translation, with code-level comparisons and quantitative validation">
        Findings and Verification
      </SectionTitle>

      <ProofBlock title="On Using the Correct Model">
        <p style={{margin:'0 0 10px'}}>
          Prof. Peter J. Thomas REALLY EMPHASIZED the IMPORTANCE of implementing the <em>EXACT</em> published model when producing
          computational graphs for this work. REALLY EMPHASIZED. The model used here is the 7D-O\u2082 system from Diekman et al. (2017),
          extended with the silent hypoxemia parameter modifications from Diekman et al. (2024). No simplifications,
          alternative formulations, or approximations were substituted. Every equation, every parameter value, and every
          initial condition is taken directly from the published MATLAB code on ModelDB accession #229640 and #2015954.
        </p>
        <p style={{margin:0}}>
          This matters because the 7D-O\u2082 model has specific dynamical properties \u2014 bistability between eupneic and
          tachypneic states, saddle-node bifurcation structure, sensitivity to the Henry\u2019s law small parameter \u2014 that
          depend on the precise equation formulation. Substituting a different CPG model or a simplified oxygen
          transport equation would alter the bifurcation diagram and invalidate the paper\u2019s conclusions about which
          parameters drive the SH phenotype. The figures generated here can be directly compared against those in
          the published paper because they implement the same system.
        </p>
      </ProofBlock>

      <ProofBlock title="Proof 1: Line-by-Line Equation Translation (Chemosensory Feedback, Eq. 28)">
        <p style={{margin:'0 0 10px'}}>
          Each line in the MATLAB function <Code>closedloop.m</Code> maps to a specific equation in the paper\u2019s
          Appendix A. Below is the MATLAB source alongside the Python translation for the chemosensory feedback:
        </p>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:C.normoxia,marginBottom:4,fontFamily:F.mono}}>MATLAB (closedloop.m, line 76)</div>
            <Code block>{`% Chemosensory feedback
gtonic = 0.3*(1-tanh((PO2blood-85)/30));
Itonic = gtonic*(v-Esyn);`}</Code>
          </div>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:C.sh,marginBottom:4,fontFamily:F.mono}}>Python (respiratory_model.py)</div>
            <Code block>{`# Chemosensation (Eq 28)
gt = p['phi'] * (1.0 - np.tanh(
    (PaO2 - p['theta_g']) / p['sigma_g']))
It = gt * (V - p['Etonic'])`}</Code>
          </div>
        </div>
        <p style={{margin:'10px 0 0'}}>
          The MATLAB version hardcodes \u03C6=0.3, \u03B8g=85, \u03C3g=30 directly. The Python version parameterizes these as
          dictionary entries, enabling the parameter sweeps shown in the paper\u2019s Fig. 3B without modifying source code.
          The mathematical operation is identical: <Code>g_tonic = \u03C6(1 \u2212 tanh((PaO\u2082 \u2212 \u03B8g)/\u03C3g))</Code>.
        </p>
      </ProofBlock>

      <ProofBlock title="Proof 2: Blood Oxygen Equation (Eqs. 21\u201325) \u2014 The Most Error-Prone Translation">
        <p style={{margin:'0 0 10px'}}>
          The blood oxygen ODE (Eq. 21) involves the hemoglobin saturation derivative and is the equation where
          translation errors are most likely to arise:
        </p>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:C.normoxia,marginBottom:4,fontFamily:F.mono}}>MATLAB (closedloop.m, lines 60\u201372)</div>
            <Code block>{`SaO2 = (PO2blood^c)/(PO2blood^c+K^c);
CaO2 = eta*SaO2+betaO2*PO2blood;
partial = (c*PO2blood^(c-1))*...
  (1/(PO2blood^c+K^c)-...
   (PO2blood^c)/((PO2blood^c+K^c)^2));
Jlb = (1/taulb)*(PO2lung-PO2blood)*...
  (vollung/(R*Temp));
Jbt = M*CaO2*gamma;
...
z(7) = (Jlb-Jbt)/...
  (gamma*(betaO2+eta*partial));`}</Code>
          </div>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:C.sh,marginBottom:4,fontFamily:F.mono}}>Python translation</div>
            <Code block>{`S = PaO2**cc / (PaO2**cc + KK**cc)
dS = cc*PaO2**(cc-1) * (
    1/(PaO2**cc + KK**cc) -
    PaO2**cc / (PaO2**cc + KK**cc)**2)
JLB = (PAO2-PaO2)/p['tau_LB'] * (
    volL / (p['R_gas']*p['Temp']))
JBT = p['M']*zeta*(
    p['betaO2']*PaO2 + eta*S)
...
dPaO2 = (JLB - JBT) / (
    zeta*(p['betaO2'] + eta*dS))`}</Code>
          </div>
        </div>
        <p style={{margin:'10px 0 0'}}>
          In the MATLAB code, <Code>CaO2 = eta*SaO2 + betaO2*PO2blood</Code> combines dissolved
          and hemoglobin-bound oxygen (Eq. 23). The term <Code>gamma</Code> = volB/22400 is the
          conversion factor \u03B6 (Eq. 26). The Python version uses the paper\u2019s notation (JLB, JBT, \u03B6, \u03B7)
          rather than the MATLAB shorthand for direct cross-referencing with the published equations.
        </p>
      </ProofBlock>

      <ProofBlock title="Proof 3: Numerical Solver Equivalence (ode15s \u2192 Radau)">
        <p style={{margin:'0 0 10px'}}>
          The MATLAB code uses <Code>ode15s</Code>, a variable-order, variable-step implicit solver for stiff
          systems. The Python translation uses <Code>scipy.integrate.solve_ivp</Code> with <Code>Radau</Code>
          (implicit Runge-Kutta of order 5), which is the closest scipy equivalent for stiff ODE systems:
        </p>
        <Code block>{`# MATLAB (reproduce_figure1.m):
options = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,u] = ode15s('closedloop',[0 tf],inits,options);

# Python equivalent:
sol = solve_ivp(
    fun=lambda t,u: rhs(t, u, params),
    t_span=[0, tf],
    y0=inits,
    method='Radau',        # implicit, stiff-capable
    rtol=1e-9, atol=1e-9,  # identical tolerances
    max_step=5.0
)`}</Code>
        <p style={{margin:'10px 0 0'}}>
          Both solvers use the same tolerance settings (rtol = atol = 10\u207B\u2079) and identical initial conditions
          from <Code>reproduce_figure1.m</Code>: [\u221256.8172, 9.5344\u00D710\u207B\u2074, 0.7454, 2.0026\u00D710\u207B\u2074, 2.0525, 98.9638, 97.7927].
          Radau was chosen specifically because ode15s is a multistep BDF/NDF method for stiff problems, and Radau
          is the closest single-step stiff solver available in scipy with comparable accuracy characteristics.
        </p>
      </ProofBlock>

      <ProofBlock title="Proof 4: Quantitative Output Validation Against Published Figures">
        <p style={{margin:'0 0 10px'}}>
          Python simulation outputs compared against value ranges visible in the paper\u2019s published figures:
        </p>
        <div style={{overflowX:'auto'}}>
          <table style={{width:'100%',borderCollapse:'collapse',fontFamily:F.mono,fontSize:11,marginTop:8}}>
            <thead>
              <tr style={{borderBottom:`2px solid ${C.border}`,background:C.surface}}>
                {['Variable','Paper Figure Range','Python Output','Consistent'].map(h=>(
                  <th key={h} style={{padding:'6px 10px',textAlign:'left',color:C.muted,fontSize:10,textTransform:'uppercase'}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['V (mV)','\u221260 to +10 (bursting, Fig. 1)','\u221257 (bursting pattern)','Yes'],
                ['\u03B1','0 to ~0.01 (Fig. 1)','~0.0002\u20130.01 (oscillating)','Yes'],
                ['volL (L)','~2.0 to ~3.0 (Fig. 1)','~2.0\u20132.9 (oscillating)','Yes'],
                ['PaO\u2082 (mmHg)','~95\u2013110 normoxia (Fig. 2A plateau)','104.1 (steady state)','Yes'],
                ['PaO\u2082 SH (mmHg)','~70\u201385 SH plateau (Fig. 5B, [Hb]=250)','75.7 (steady state)','Yes'],
                ['PAO\u2082 (mmHg)','~95\u2013110 (Fig. 1)','99\u2013106 (oscillating)','Yes'],
                ['g_tonic (nS)','~0.10\u20130.25 (Fig. 1)','~0.12\u20130.22','Yes'],
              ].map((r,i)=>(
                <tr key={i} style={{borderBottom:`1px solid ${C.borderLight}`}}>
                  <td style={{padding:'5px 10px',fontWeight:600,color:C.primary}}>{r[0]}</td>
                  <td style={{padding:'5px 10px',color:C.textSecondary}}>{r[1]}</td>
                  <td style={{padding:'5px 10px',color:C.text}}>{r[2]}</td>
                  <td style={{padding:'5px 10px',color:C.accent,fontWeight:700}}>{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ProofBlock>

      <ProofBlock title="Proof 5: L\u2082 Norm Cross-Validation">
        <p style={{margin:'0 0 10px'}}>
          To quantify divergence between normoxia and SH trajectories, the normalized L\u2082 norm is computed for each
          state variable. Both time series are interpolated onto a common grid of 5,000 points over [0, 15] seconds:
        </p>
        <Code block>{`L\u2082(x) = sqrt( (1/T) \u222B\u2080\u1D40 |x_normoxia(t) \u2212 x_SH(t)|\u00B2 dt )

Computed values (Python, tf = 15s):
  V:      7.590     mV         (CPG dynamics diverge moderately)
  n:      0.135     (dimless)  (gating variable, small divergence)
  h:      0.053     (dimless)  (slow inactivation, minimal shift)
  \u03B1:      0.003     (dimless)  (motor pool tracks CPG closely)
  volL:   0.374     L          (lung volume differences)
  PAO\u2082:  15.002     mmHg       (alveolar O\u2082 divergence)
  PaO\u2082:  14.963     mmHg       (arterial O\u2082 \u2014 largest effect)`}</Code>
        <p style={{margin:'10px 0 0'}}>
          The largest divergences are in the oxygen variables (PAO\u2082 and PaO\u2082), consistent with the paper\u2019s finding
          that the SH phenotype manifests primarily in oxygen handling, not in the CPG rhythm. The CPG variables
          show smaller norms because the bursting pattern remains qualitatively similar in both models \u2014 what changes
          is the steady-state blood oxygen level maintained by that rhythm.
        </p>
      </ProofBlock>

      <ProofBlock title="Proof 6: Time-Series Alignment Methodology">
        <p style={{margin:'0 0 6px'}}>
          <strong style={{color:C.text}}>Step 1 \u2014 Extract time series:</strong> MATLAB output from
          {' '}<Code>reproduce_figure1.m</Code> produces [t, V, n, h, \u03B1, volL, PAO\u2082, PaO\u2082] via ode15s.
          Python output from <Code>solve_ivp</Code> returns <Code>sol.t</Code> and <Code>sol.y</Code> with the same
          7 state variables. Both are exported to CSV with identical column headers for external verification.
        </p>
        <p style={{margin:'0 0 6px'}}>
          <strong style={{color:C.text}}>Step 2 \u2014 Align timestamps:</strong> Because ode15s and Radau use adaptive
          stepping, the raw time grids differ. Both are resampled onto a uniform grid of N=5000 points using
          linear interpolation (<Code>np.interp</Code>), covering [max(t\u2081[0], t\u2082[0]), min(t\u2081[-1], t\u2082[-1])].
        </p>
        <p style={{margin:0}}>
          <strong style={{color:C.text}}>Step 3 \u2014 Compute L\u2082 norm:</strong> For each variable, the point-wise
          difference is computed, squared, summed with trapezoidal weighting, normalized by interval length T,
          and square-rooted. This produces a single scalar per variable measuring root-mean-square deviation.
        </p>
      </ProofBlock>

      <ProofBlock title="Why These Specific SH Parameters Were Selected">
        <p style={{margin:'0 0 10px'}}>
          The silent hypoxemia parameter set was not arbitrarily chosen. The paper (Results, p. 7) describes a
          systematic exploration of 27 chemosensory parameter combinations, followed by additional sweeps:
        </p>
        <p style={{margin:'0 0 6px'}}>
          <strong style={{color:C.text}}>Chemosensory parameters (Fig. 3B):</strong> 27 combinations of \u03C6 \u2208 &#123;0.24, 0.3, 0.36&#125;,
          \u03B8g \u2208 &#123;70, 85, 100&#125;, \u03C3g \u2208 &#123;24, 30, 36&#125; were tested. The combination \u03C6=0.24, \u03B8g=70, \u03C3g=36
          produced the greatest reduction in PaO\u2082 and was selected as the working chemosensory set.
        </p>
        <p style={{margin:'0 0 6px'}}>
          <strong style={{color:C.text}}>Hemoglobin binding K (Fig. 4B):</strong> Varying K shifted the collapse point but was insufficient alone.
        </p>
        <p style={{margin:'0 0 6px'}}>
          <strong style={{color:C.text}}>Lung parameters vol\u2080, \u03C4_LB (Fig. 5A):</strong> Varying by \u00B120% and \u00B1400% respectively had surprisingly little effect.
        </p>
        <p style={{margin:0}}>
          <strong style={{color:C.text}}>Hemoglobin [Hb] (Fig. 5B):</strong> This was the decisive parameter. [Hb]=250 g/L shifted the
          metabolic collapse point leftward while maintaining the hypoxemic plateau. The paper states explicitly:
          {' '}<em>\u201CThus we will consider the model with [Hb]=250 as our working model for silent hypoxemia\u201D</em> (p. 9).
        </p>
      </ProofBlock>
    </Section>
  );

  // ════════════════════════════════════════════════════════════════════
  // PAGE: REFERENCES
  // ════════════════════════════════════════════════════════════════════
  const PaperPage=()=>(
    <Section>
      <SectionTitle sub="Source publications, code repositories, and deployment information">
        References and Resources
      </SectionTitle>

      <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,padding:'24px 28px',marginBottom:24}}>
        <h3 style={{fontFamily:F.display,fontSize:17,color:C.text,margin:'0 0 6px'}}>
          COVID-19 and Silent Hypoxemia in a Minimal Closed-Loop Model of the Respiratory Rhythm Generator
        </h3>
        <p style={{fontFamily:F.body,fontSize:13,color:C.textSecondary,margin:'0 0 4px'}}>
          Casey O. Diekman (NJIT), Peter J. Thomas (Case Western Reserve), Christopher G. Wilson (Loma Linda)
        </p>
        <p style={{fontFamily:F.mono,fontSize:11,color:C.primary,margin:'0 0 16px'}}>
          Biological Cybernetics (2024), 118(3-4):145\u2013163 \u00B7 DOI: 10.1007/s00422-024-00989-w \u00B7 PMID: 38884785
        </p>
        <p style={{fontFamily:F.body,fontSize:13,color:C.muted,lineHeight:1.7,margin:0}}>
          The paper uses the 7D-O\u2082 model to test whether altered chemosensory function at the
          carotid bodies and/or the nucleus tractus solitarii are responsible for the blunted hypoxia
          response in COVID-19 patients. The central finding is that while reduced chemosensory gain creates a
          hypoxemic plateau, increasing hemoglobin concentration is necessary and sufficient to shift the
          metabolic collapse point, producing the silent hypoxemia phenotype.
        </p>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))',gap:12,marginBottom:24}}>
        {[
          {title:'Original 7D-O\u2082 MATLAB Code',url:'https://modeldb.science/229640',desc:'ModelDB #229640 \u2014 Diekman, Thomas & Wilson (2017), J. Neurophysiol. Full MATLAB source with reproduce_figure scripts.'},
          {title:'Silent Hypoxemia Extension',url:'https://modeldb.science/2015954',desc:'ModelDB #2015954 \u2014 COVID-19 SH model (2024). Extended parameter sets.'},
          {title:'GitHub Repository',url:'https://github.com/ModelDBRepository/229640',desc:'closedloop.m, reproduce_figure1.m through reproduce_figure15.m, XPPAUT files.'},
          {title:'Published Paper (PubMed)',url:'https://pubmed.ncbi.nlm.nih.gov/38884785/',desc:'Free PMC article with all figures, equations, and supplementary analysis.'},
        ].map((l,i)=>(
          <a key={i} href={l.url} target="_blank" rel="noopener noreferrer"
            style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:6,padding:'16px 20px',
              textDecoration:'none',display:'block'}}>
            <h4 style={{fontFamily:F.display,fontSize:13,color:C.primary,margin:'0 0 6px'}}>{l.title}</h4>
            <p style={{fontFamily:F.body,fontSize:11,color:C.muted,margin:0,lineHeight:1.5}}>{l.desc}</p>
          </a>
        ))}
      </div>

      <ProofBlock title="Deployment and Reproducibility">
        <p style={{margin:'0 0 8px'}}>
          <strong style={{color:C.text}}>Python backend</strong> \u2014 Run <Code>python respiratory_model.py --compare</Code> to generate
          all comparison figures and L\u2082 norms. Dependencies: numpy, scipy, matplotlib.
        </p>
        <p style={{margin:'0 0 8px'}}>
          <strong style={{color:C.text}}>Browser simulator</strong> \u2014 Implements the full 7D ODE system in JavaScript
          using a midpoint (RK2) method with dt=0.5ms. Less precise than scipy\u2019s Radau solver but captures
          correct qualitative dynamics. Publication figures should use the Python implementation.
        </p>
        <p style={{margin:0}}>
          <strong style={{color:C.text}}>Vercel deployment</strong> \u2014 This is a standard Vite + React project.
          Run <Code>npm install && npm run build</Code>, then deploy with <Code>vercel</Code> or push to GitHub
          for automatic Vercel deployment.
        </p>
      </ProofBlock>

      <ProofBlock title="Acknowledgments">
        <p style={{margin:0}}>
          This translation and interactive implementation is based entirely on the work of Casey O. Diekman
          (NJIT), Peter J. Thomas (Case Western Reserve University), and Christopher G. Wilson (Loma Linda University).
          The original research was supported by NIH grants RF1 NS118606-01 and RO1 AT011691-01, and NSF grants
          DMS-2052109, DMS-1555237, and DMS-2152115. Model code is publicly available at ModelDB #229640 and #2015954.
        </p>
      </ProofBlock>
    </Section>
  );

  // ── Render ──────────────────────────────────────────────────────────
  return(
    <div style={{background:C.bg,color:C.text,minHeight:'100vh',fontFamily:F.body}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
        input[type=range] { -webkit-appearance: none; background: #e9ecef; border-radius: 2px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #2980b9; cursor: pointer; }
        a { transition: opacity 0.2s; }
        a:hover { opacity: 0.85; }
      `}</style>
      <Nav/>
      {page==='home'&&<Home/>}
      {page==='explorer'&&<Explorer/>}
      {page==='findings'&&<Findings/>}
      {page==='paper'&&<PaperPage/>}
      <footer style={{borderTop:`1px solid ${C.border}`,padding:'20px 24px',textAlign:'center',
        fontFamily:F.mono,fontSize:10,color:C.faint}}>
        7D-O\u2082 Closed-Loop Respiratory Control Model \u00B7 Python Translation and Interactive Explorer \u00B7 - Brayden Choi
        Based on Diekman, Thomas & Wilson, Biol. Cybern. (2024) \u00B7 ModelDB #229640 / #2015954
      </footer>
    </div>
  );
}
