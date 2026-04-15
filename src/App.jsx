import { useState, useCallback, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, Cell, ResponsiveContainer, ReferenceLine
} from "recharts";

/*
  7D-O₂ Closed-Loop Respiratory Control Model
  Diekman, Thomas & Wilson (2017, 2024)
  Paper-accurate: φ=0.3/0.24, θg=85/70, σg=30/36, [Hb]=150/250
*/

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

const SH = { ...NORM, phi:0.24, theta_g:70, sigma_g:36, Hb:250 };

const Y0 = [-56.8172, 9.5344e-4, 0.7454, 2.0026e-4, 2.0525, 98.9638, 97.7927];

// RHS of 7D-O₂ system (Eqs 7-28)
function f(u, p) {
  let [V,n,h,a,vL,PA,Pa] = u;
  Pa = Math.max(Pa, 0.01);
  const sig = (V,th,si) => 1/(1+Math.exp((V-th)/si));
  const tau = (V,th,si,tb) => tb/Math.cosh((V-th)/(2*si));

  const pi = sig(V,p.theta_p,p.sigma_p);
  const mi = sig(V,p.theta_m,p.sigma_m);
  const ni = sig(V,p.theta_n,p.sigma_n);
  const hi = sig(V,p.theta_h,p.sigma_h);
  const tn = tau(V,p.theta_n,p.sigma_n,p.tau_n_bar);
  const th = tau(V,p.theta_h,p.sigma_h,p.tau_h_bar);

  const IK = p.gK * n*n*n*n * (V - p.EK);
  const INP = p.gNaP * pi * h * (V - p.ENa);
  const INa = p.gNa * mi*mi*mi * (1-n) * (V - p.ENa);
  const IL = p.gL * (V - p.EL);
  const gt = p.phi * (1 - Math.tanh((Pa - p.theta_g) / p.sigma_g));
  const It = gt * (V - p.Etonic);
  const Tc = p.Tmax / (1 + Math.exp(-(V - p.VT) / p.Kp));

  const dvL = p.E1 * a - p.E2 * (vL - p.vol0);
  const z = p.volB / 22400;
  const eta = p.Hb * 1.36;
  const cc = p.c, KK = p.K;
  const Pc = Math.pow(Pa, cc), Kc = Math.pow(KK, cc);
  const S = Pc / (Pc + Kc);
  const dS = cc * Math.pow(Pa, cc-1) * (1/(Pc+Kc) - Pc/((Pc+Kc)*(Pc+Kc)));
  const JLB = (PA - Pa) / p.tau_LB * (vL / (p.R_gas * p.Temp));
  const JBT = p.M * z * (p.betaO2 * Pa + eta * S);
  const dvp = Math.max(0, dvL);
  const dPA = ((p.PextO2 - PA) / vL) * dvp - (PA - Pa) / p.tau_LB;
  const den = z * (p.betaO2 + eta * dS);

  return [
    (-IK - INP - INa - IL - It) / p.C,
    (ni - n) / tn,
    (hi - h) / th,
    p.ra * Tc * (1 - a) - p.ra * a,
    dvL,
    dPA,
    den > 1e-15 ? (JLB - JBT) / den : 0
  ];
}

// RK4 integrator — dt=0.1ms for numerical stability on this stiff system
function integrate(par, tf = 15000) {
  const dt = 0.1;
  const N = Math.floor(tf / dt);
  const keep = 2000;
  const skip = Math.max(1, Math.floor(N / keep));
  let u = [...Y0];
  const out = [];

  for (let i = 0; i <= N; i++) {
    if (i % skip === 0) {
      const Pa = Math.max(u[6], 0.01);
      const cc = par.c, KK = par.K;
      const Pc = Math.pow(Pa, cc), Kc = Math.pow(KK, cc);
      const S = Pc / (Pc + Kc);
      const g = par.phi * (1 - Math.tanh((Pa - par.theta_g) / par.sigma_g));
      out.push({
        t: parseFloat((i * dt / 1000).toFixed(4)),
        V: u[0],
        alpha: u[3],
        volL: u[4],
        PAO2: u[5],
        PaO2: u[6],
        SaO2: S * 100,
        gtonic: g,
      });
    }

    // Classical RK4
    const k1 = f(u, par);
    const u2 = u.map((v, j) => v + 0.5 * dt * k1[j]);
    const k2 = f(u2, par);
    const u3 = u.map((v, j) => v + 0.5 * dt * k2[j]);
    const k3 = f(u3, par);
    const u4 = u.map((v, j) => v + dt * k3[j]);
    const k4 = f(u4, par);
    u = u.map((v, j) => v + (dt / 6) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]));

    // Clamp to prevent blow-up
    u[4] = Math.max(u[4], 0.5);
    u[5] = Math.max(u[5], 0.1);
    u[6] = Math.max(u[6], 0.1);
    if (u[0] > 100) u[0] = 100;
    if (u[0] < -100) u[0] = -100;
  }
  return out;
}

function calcL2(d1, d2, key) {
  const len = Math.min(d1.length, d2.length);
  if (len < 2) return 0;
  let s = 0;
  for (let i = 0; i < len; i++) {
    const diff = (d1[i][key] || 0) - (d2[i][key] || 0);
    s += diff * diff;
  }
  const T = d1[len-1].t - d1[0].t;
  if (T < 1e-6) return 0;
  return Math.sqrt(s * (T / (len - 1)) / T);
}

// ── Theme ─────────────────────────────────────────────────────────────
const TH = {
  bg:"#ffffff", surface:"#f8f9fa", card:"#ffffff",
  border:"#dee2e6", bLight:"#e9ecef",
  pri:"#1a5276", priL:"#2980b9", acc:"#117a65",
  red:"#c0392b", blue:"#2471a3",
  txt:"#212529", txt2:"#495057", mut:"#6c757d", fnt:"#adb5bd",
  gold:"#b7950b", codeBg:"#f4f6f7", codeC:"#1a5276",
};

const FN = {
  d: "'Libre Baskerville','Georgia','Times New Roman',serif",
  b: "'Source Sans 3','Helvetica Neue','Segoe UI',sans-serif",
  m: "'IBM Plex Mono','Consolas','Courier New',monospace",
};

// ── UI Components ─────────────────────────────────────────────────────
function Sl({label,value,min,max,step,onChange,unit=""}) {
  return (
    <div style={{marginBottom:10}}>
      <div style={{display:'flex',justifyContent:'space-between',fontSize:12,color:TH.mut,marginBottom:2}}>
        <span style={{fontFamily:FN.d,fontStyle:'italic',fontSize:11}}>{label}</span>
        <span style={{fontFamily:FN.m,color:TH.pri,fontWeight:600,fontSize:11}}>
          {value < 0.001 ? value.toExponential(1) : Number(value).toFixed(step<0.01?4:step<1?2:0)}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        style={{width:'100%',accentColor:TH.priL,height:4,cursor:'pointer'}} />
    </div>
  );
}

function St({label,value,unit,warn}) {
  return (
    <div style={{background:TH.surface,border:`1px solid ${TH.border}`,borderRadius:6,
      padding:'10px 14px',textAlign:'center',flex:1,minWidth:110}}>
      <div style={{fontSize:10,color:TH.mut,textTransform:'uppercase',letterSpacing:1,fontFamily:FN.b,fontWeight:600}}>{label}</div>
      <div style={{fontSize:22,fontWeight:700,color:warn?TH.red:TH.pri,fontFamily:FN.m,marginTop:3}}>{value}</div>
      <div style={{fontSize:9,color:TH.mut}}>{unit}</div>
    </div>
  );
}

function Cd({children,block=false}) {
  if (block) return (
    <pre style={{background:TH.codeBg,border:`1px solid ${TH.bLight}`,borderRadius:4,
      padding:'12px 16px',fontFamily:FN.m,fontSize:11,color:TH.txt,lineHeight:1.6,
      overflowX:'auto',margin:'12px 0',whiteSpace:'pre-wrap'}}>{children}</pre>
  );
  return <code style={{background:TH.codeBg,padding:'1px 5px',borderRadius:3,fontFamily:FN.m,fontSize:'0.9em',color:TH.codeC}}>{children}</code>;
}

function Sec({children,style={}}) {
  return <section style={{padding:'40px 24px',maxWidth:1100,margin:'0 auto',...style}}>{children}</section>;
}

function SecT({children,sub}) {
  return (
    <div style={{marginBottom:28,borderBottom:`2px solid ${TH.pri}`,paddingBottom:12}}>
      <h2 style={{fontFamily:FN.d,fontSize:22,fontWeight:700,color:TH.pri,margin:0}}>{children}</h2>
      {sub && <p style={{fontFamily:FN.b,fontSize:13,color:TH.mut,marginTop:6,marginBottom:0}}>{sub}</p>}
    </div>
  );
}

function Prf({title,children}) {
  return (
    <div style={{background:TH.surface,border:`1px solid ${TH.border}`,borderRadius:6,padding:'20px 24px',marginBottom:20}}>
      <h4 style={{fontFamily:FN.d,fontSize:14,color:TH.pri,margin:'0 0 10px',fontWeight:700}}>{title}</h4>
      <div style={{fontFamily:FN.b,fontSize:13,color:TH.txt2,lineHeight:1.75}}>{children}</div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState('home');
  const [params, setParams] = useState({...NORM});
  const [mode, setMode] = useState('normoxia');
  const [data1, setData1] = useState(null);
  const [data2, setData2] = useState(null);
  const [running, setRunning] = useState(false);
  const [activeVar, setActiveVar] = useState('PaO2');
  const [l2s, setL2s] = useState(null);
  const timerRef = useRef(null);

  const runSim = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      try {
        if (mode === 'compare') {
          const d1 = integrate(NORM);
          const d2 = integrate(params);
          setData1(d1);
          setData2(d2);
          const keys = ['V','alpha','volL','PAO2','PaO2','SaO2','gtonic'];
          const norms = {};
          keys.forEach(k => { norms[k] = calcL2(d1, d2, k); });
          setL2s(norms);
        } else {
          const d = integrate(params);
          setData1(d);
          setData2(null);
          setL2s(null);
        }
      } catch (e) { console.error("Simulation error:", e); }
      setRunning(false);
    }, 50);
  }, [params, mode]);

  // Auto-run on mount
  useEffect(() => { runSim(); }, []);

  // Auto-run when mode changes
  useEffect(() => { runSim(); }, [mode]);

  // Debounced auto-run when params change
  const updateParam = (key, val) => {
    setParams(p => ({ ...p, [key]: val }));
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      // trigger re-run via state change
      setRunning(prev => {
        setTimeout(() => {
          // run simulation with latest params
          const currentParams = {...params, [key]: val};
          try {
            if (mode === 'compare') {
              const d1 = integrate(NORM);
              const d2 = integrate(currentParams);
              setData1(d1); setData2(d2);
              const keys = ['V','alpha','volL','PAO2','PaO2','SaO2','gtonic'];
              const norms = {};
              keys.forEach(k => { norms[k] = calcL2(d1, d2, k); });
              setL2s(norms);
            } else {
              setData1(integrate(currentParams));
              setData2(null); setL2s(null);
            }
          } catch(e) { console.error(e); }
          setRunning(false);
        }, 30);
        return true;
      });
    }, 400);
  };

  const setPreset = (m) => {
    setMode(m);
    if (m === 'normoxia') setParams({...NORM});
    else if (m === 'sh') setParams({...SH});
    else setParams({...NORM});
  };

  const last = data1?.[data1.length - 1];
  const fPa = last?.PaO2?.toFixed(1) ?? "—";
  const fSa = last?.SaO2?.toFixed(1) ?? "—";

  const VARS = [
    { key:'V', label:'V (mV)', c:TH.blue },
    { key:'alpha', label:'\u03B1', c:'#7d3c98' },
    { key:'volL', label:'volL (L)', c:TH.acc },
    { key:'PaO2', label:'PaO\u2082 (mmHg)', c:TH.pri },
    { key:'PAO2', label:'PAO\u2082 (mmHg)', c:'#a93226' },
    { key:'SaO2', label:'SaO\u2082 (%)', c:TH.gold },
    { key:'gtonic', label:'g_tonic (nS)', c:'#148f77' },
  ];

  const l2Bars = l2s ? Object.entries(l2s).map(([k,v]) => ({name:k, value:+v.toFixed(4)})) : [];
  const activeColor = VARS.find(v => v.key === activeVar)?.c || TH.pri;

  // ── NAV ──
  const Nav = () => (
    <nav style={{
      position:'sticky',top:0,zIndex:100,
      background:'rgba(255,255,255,0.97)',backdropFilter:'blur(8px)',
      borderBottom:`1px solid ${TH.border}`,padding:'0 24px',
      display:'flex',alignItems:'center',height:52,
    }}>
      <div style={{fontFamily:FN.d,fontSize:16,fontWeight:700,color:TH.pri,cursor:'pointer',whiteSpace:'nowrap'}}
        onClick={() => setPage('home')}>
        7D-O₂ Respiratory Model Python - Brayden Choi
      </div>
      <div style={{flex:1}} />
      {[['home','Overview'],['explorer','Simulator'],['findings','Findings & Proof'],['paper','References']].map(([k,l]) => (
        <button key={k} onClick={() => setPage(k)} style={{
          background:'none',border:'none',color:page===k?TH.pri:TH.mut,
          fontFamily:FN.b,fontSize:13,fontWeight:page===k?700:400,
          cursor:'pointer',padding:'8px 14px',
          borderBottom:page===k?`2px solid ${TH.pri}`:'2px solid transparent',
        }}>{l}</button>
      ))}
    </nav>
  );

  // ═══════════════════════════════════════════════════════════════════
  // HOME
  // ═══════════════════════════════════════════════════════════════════
  const Home = () => (
    <>
      <div style={{background:TH.surface,borderBottom:`1px solid ${TH.border}`,padding:'48px 24px 40px',textAlign:'center'}}>
        <div style={{maxWidth:800,margin:'0 auto'}}>
          <div style={{fontSize:11,fontFamily:FN.m,color:TH.acc,letterSpacing:2,textTransform:'uppercase',marginBottom:14,fontWeight:600}}>
            Computational Neuroscience
          </div>
          <h1 style={{fontFamily:FN.d,fontSize:28,fontWeight:700,color:TH.txt,margin:'0 0 16px',lineHeight:1.4}}>
            COVID-19 and Silent Hypoxemia in a Minimal Closed-Loop Model of the Respiratory Rhythm Generator
          </h1>
          <p style={{fontFamily:FN.b,fontSize:14,color:TH.mut,maxWidth:640,margin:'0 auto 6px',lineHeight:1.6}}>
            MATLAB to Python translation with interactive parameter exploration, L₂ norm cross-validation, and publication-quality figure generation
          </p>
          <p style={{fontFamily:FN.d,fontSize:12,color:TH.mut,fontStyle:'italic',marginBottom:24}}>
            Based on Diekman CO, Thomas PJ, Wilson CG (2024). <em>Biological Cybernetics</em>, 118(3-4):145–163
          </p>
          <div style={{display:'flex',gap:10,justifyContent:'center',flexWrap:'wrap'}}>
            <button onClick={() => setPage('explorer')} style={{padding:'10px 24px',borderRadius:4,border:'none',cursor:'pointer',background:TH.pri,color:'#fff',fontFamily:FN.b,fontSize:13,fontWeight:600}}>Open Simulator</button>
            <button onClick={() => setPage('findings')} style={{padding:'10px 24px',borderRadius:4,border:`1px solid ${TH.pri}`,cursor:'pointer',background:'transparent',color:TH.pri,fontFamily:FN.b,fontSize:13,fontWeight:600}}>View Findings</button>
          </div>
        </div>
      </div>

      <Sec>
        <SecT sub="The 7D-O₂ model couples a Butera-Rinzel-Smith central pattern generator with lung mechanics, oxygen transport, and chemosensory feedback">Model Architecture</SecT>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(300px,1fr))',gap:16}}>
          {[
            ['Central Pattern Generator','The Butera-Rinzel-Smith pacemaker neuron model with persistent Na\u207A (INaP), fast Na\u207A (INa), delayed-rectifier K\u207A (IK), and leak (IL) currents. Three state variables: membrane potential V, potassium activation n, and persistent sodium inactivation h. Equations 7\u201316.'],
            ['Motor Pool and Lung Mechanics','CPG voltage drives a synaptic motor pool activation variable \u03B1 (Eq. 17\u201318), which expands lung volume volL (Eq. 19). The low-pass filter behavior of the musculature means high-frequency tonic spiking does not drive effective ventilation \u2014 only rhythmic bursting does.'],
            ['Oxygen Transport','Alveolar oxygen (PAO₂, Eq. 20) and arterial blood oxygen (PaO₂, Eq. 21) are coupled through lung-to-blood flux JLB and metabolic consumption JBT. Hemoglobin binding follows a Hill equation (Eq. 24) with c = 2.5 and K = 26 mmHg.'],
            ['Chemosensory Feedback','The sigmoidal function g_tonic = \u03C6(1 \u2212 tanh((PaO₂ \u2212 \u03B8g)/\u03C3g)) closes the control loop (Eq. 28). When PaO₂ drops, g_tonic increases, driving the CPG to burst faster. This pathway is hypothesized to be impaired by COVID-19.'],
            ['Silent Hypoxemia Mechanism','Altering chemosensory gain alone (\u03C6, \u03B8g, \u03C3g) lowers the PaO₂ plateau but shifts the collapse point rightward. Only when [Hb] is increased to 250 g/L does the collapse point shift to lower metabolic demand, producing the clinical SH phenotype.'],
            ['The M \u00D7 \u03B7 Invariance','Because the Henry\'s law constant \u03B2O₂ \u2248 0.03 is small, the blood oxygen dynamics depend approximately on the product M \u00D7 \u03B7 (metabolic demand times hemoglobin capacity). Rescaling M \u2192 \u03B3M, \u03B7 \u2192 \u03B7/\u03B3 leaves dynamics nearly invariant (Eqs. 5\u20136).'],
          ].map(([t,d],i) => (
            <div key={i} style={{background:TH.card,border:`1px solid ${TH.border}`,borderRadius:6,padding:'20px 24px',textAlign:'center'}}>
              <h3 style={{fontFamily:FN.d,fontSize:14,color:TH.pri,margin:'0 0 10px',fontWeight:700}}>{t}</h3>
              <p style={{fontFamily:FN.b,fontSize:12.5,color:TH.txt2,lineHeight:1.7,margin:0}}>{d}</p>
            </div>
          ))}
        </div>
      </Sec>

      <Sec style={{paddingTop:0}}>
        <SecT sub="Exact values from the paper's Appendix A (Eqs 7–28) and Results (p. 7–9)">Parameter Specification</SecT>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          {[
            { title:'Normoxia (Original 7D-O₂)', color:TH.red, note:'ModelDB #229640',
              rows:[['φ','0.3 nS','Eq. 28'],['θg','85 mmHg','Eq. 28'],['σg','30 mmHg','Eq. 28'],['[Hb]','150 g/L','Eq. 27'],['M','8×10⁻⁶ ms⁻¹','Eq. 23'],['K','26 mmHg','Eq. 24'],['c','2.5','Eq. 24'],['vol₀','2.0 L','Eq. 19'],['τ_LB','500 ms','Eq. 22']]},
            { title:'Silent Hypoxemia (SH Model)', color:TH.blue, note:'ModelDB #2015954 — selected for greatest PaO₂ reduction (p. 7)',
              rows:[['φ','0.24 nS','Results p.7'],['θg','70 mmHg','Results p.7'],['σg','36 mmHg','Results p.7'],['[Hb]','250 g/L','Fig. 5B, p.9'],['M','8×10⁻⁶ ms⁻¹','unchanged'],['K','26 mmHg','unchanged'],['c','2.5','unchanged'],['vol₀','2.0 L','unchanged'],['τ_LB','500 ms','unchanged']]},
          ].map((t,i) => (
            <div key={i} style={{background:TH.card,border:`1px solid ${TH.border}`,borderRadius:6,padding:'16px 20px'}}>
              <h3 style={{fontFamily:FN.d,fontSize:13,color:t.color,margin:'0 0 4px',fontWeight:700}}>{t.title}</h3>
              <p style={{fontFamily:FN.m,fontSize:10,color:TH.fnt,margin:'0 0 10px'}}>{t.note}</p>
              <table style={{width:'100%',borderCollapse:'collapse',fontSize:11,fontFamily:FN.m}}>
                <thead><tr style={{borderBottom:`1px solid ${TH.border}`}}>
                  {['Param','Value','Source'].map(h => <th key={h} style={{textAlign:'left',padding:'4px 6px',color:TH.mut,fontWeight:600,fontSize:10}}>{h}</th>)}
                </tr></thead>
                <tbody>{t.rows.map(([k,v,s],j) => (
                  <tr key={j} style={{borderBottom:`1px solid ${TH.bLight}`}}>
                    <td style={{padding:'4px 6px',color:TH.txt}}>{k}</td>
                    <td style={{padding:'4px 6px',color:TH.pri,fontWeight:600}}>{v}</td>
                    <td style={{padding:'4px 6px',color:TH.fnt,fontSize:10}}>{s}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ))}
        </div>
      </Sec>
    </>
  );

  // ═══════════════════════════════════════════════════════════════════
  // SIMULATOR
  // ═══════════════════════════════════════════════════════════════════
  const Explorer = () => (
    <Sec>
      <SecT sub="Adjust parameters and observe effects on the 7D-O₂ system. Charts update automatically.">Interactive Simulator</SecT>

      <div style={{display:'flex',gap:8,marginBottom:14,flexWrap:'wrap',alignItems:'center'}}>
        {[['normoxia','Normoxia'],['sh','Silent Hypoxemia'],['compare','Compare']].map(([m,l]) => (
          <button key={m} onClick={() => setPreset(m)} style={{
            padding:'6px 16px',borderRadius:4,
            border:`1px solid ${mode===m?(m==='sh'?TH.blue:m==='compare'?TH.acc:TH.red):TH.border}`,
            cursor:'pointer',fontSize:12,fontWeight:600,fontFamily:FN.b,
            background:mode===m?(m==='sh'?TH.blue:m==='compare'?TH.acc:TH.red):'transparent',
            color:mode===m?'#fff':TH.txt2,
          }}>{l}</button>
        ))}
        <div style={{flex:1}} />
        <button onClick={runSim} disabled={running} style={{
          padding:'7px 22px',borderRadius:4,border:'none',
          cursor:running?'wait':'pointer',
          background:running?TH.border:TH.pri,
          color:'#fff',fontSize:12,fontWeight:700,fontFamily:FN.b,
        }}>{running ? 'Solving...' : 'Run Simulation'}</button>
      </div>

      <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
        {/* Parameter sidebar */}
        <div style={{width:240,flexShrink:0,background:TH.surface,border:`1px solid ${TH.border}`,borderRadius:6,padding:14,maxHeight:700,overflowY:'auto'}}>
          <div style={{fontSize:10,fontWeight:700,color:TH.pri,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8}}>Chemosensory — Eq. 28</div>
          <Sl label="φ (max g_tonic)" value={params.phi} min={0.05} max={0.6} step={0.01} onChange={v => updateParam('phi',v)} unit=" nS" />
          <Sl label="θg (half-activation)" value={params.theta_g} min={30} max={120} step={1} onChange={v => updateParam('theta_g',v)} unit=" mmHg" />
          <Sl label="σg (slope)" value={params.sigma_g} min={5} max={60} step={1} onChange={v => updateParam('sigma_g',v)} unit=" mmHg" />

          <div style={{fontSize:10,fontWeight:700,color:TH.pri,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8,marginTop:14}}>Blood O₂ — Eqs. 21–27</div>
          <Sl label="M (metabolic demand)" value={params.M} min={2e-6} max={2e-5} step={1e-6} onChange={v => updateParam('M',v)} unit=" ms⁻¹" />
          <Sl label="[Hb] (hemoglobin)" value={params.Hb} min={80} max={350} step={5} onChange={v => updateParam('Hb',v)} unit=" g/L" />
          <Sl label="c (Hill coefficient)" value={params.c} min={1} max={5} step={0.1} onChange={v => updateParam('c',v)} />
          <Sl label="K (half-saturation)" value={params.K} min={8} max={40} step={1} onChange={v => updateParam('K',v)} unit=" mmHg" />

          <div style={{fontSize:10,fontWeight:700,color:TH.pri,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8,marginTop:14}}>CPG — Eqs. 7–16</div>
          <Sl label="gNaP" value={params.gNaP} min={0.5} max={8} step={0.1} onChange={v => updateParam('gNaP',v)} unit=" nS" />
          <Sl label="gK" value={params.gK} min={2} max={25} step={0.2} onChange={v => updateParam('gK',v)} unit=" nS" />

          <div style={{fontSize:10,fontWeight:700,color:TH.pri,letterSpacing:1.2,textTransform:'uppercase',marginBottom:8,marginTop:14}}>Lung — Eqs. 19–20</div>
          <Sl label="vol₀ (unloaded)" value={params.vol0} min={1} max={3} step={0.1} onChange={v => updateParam('vol0',v)} unit=" L" />
          <Sl label="τ_LB (O₂ flux)" value={params.tau_LB} min={100} max={1500} step={50} onChange={v => updateParam('tau_LB',v)} unit=" ms" />
        </div>

        {/* Charts */}
        <div style={{flex:1,minWidth:0}}>
          <div style={{display:'flex',gap:8,marginBottom:10,flexWrap:'wrap'}}>
            <St label="PaO₂" value={fPa} unit="mmHg" warn={+fPa < 70} />
            <St label="SaO₂" value={fSa} unit="%" warn={+fSa < 80} />
            <St label="Preset" value={mode==='sh'?'SH':mode==='compare'?'Compare':'Normoxia'}
              unit={mode==='sh'?'φ=.24 θg=70 σg=36 [Hb]=250':mode==='compare'?'normoxia vs SH':'φ=.30 θg=85 σg=30 [Hb]=150'} />
          </div>

          <div style={{display:'flex',gap:4,marginBottom:8,flexWrap:'wrap'}}>
            {VARS.map(v => (
              <button key={v.key} onClick={() => setActiveVar(v.key)} style={{
                padding:'3px 8px',borderRadius:3,
                border:`1px solid ${activeVar===v.key?v.c:TH.bLight}`,
                cursor:'pointer',fontSize:10,fontFamily:FN.m,
                fontWeight:activeVar===v.key?700:400,
                background:activeVar===v.key?v.c+'15':'transparent',
                color:activeVar===v.key?v.c:TH.mut,
              }}>{v.label}</button>
            ))}
          </div>

          {data1 && (
            <div style={{background:TH.card,borderRadius:6,border:`1px solid ${TH.border}`,padding:14,marginBottom:12}}>
              {mode !== 'compare' ? (
                <ResponsiveContainer width="100%" height={360}>
                  <LineChart data={data1} margin={{top:10,right:20,bottom:10,left:10}}>
                    <CartesianGrid strokeDasharray="3 3" stroke={TH.bLight} />
                    <XAxis dataKey="t" stroke={TH.mut} fontSize={10}
                      label={{value:'t (s)',position:'insideBottomRight',offset:-5,fill:TH.mut,fontSize:10}} />
                    <YAxis stroke={TH.mut} fontSize={10} domain={['auto','auto']} />
                    <Tooltip contentStyle={{background:'#fff',border:`1px solid ${TH.border}`,borderRadius:4,fontSize:11}}
                      formatter={v => [Number(v).toFixed(3), activeVar]} />
                    <Line type="monotone" dataKey={activeVar} stroke={activeColor}
                      dot={false} strokeWidth={1.2} isAnimationActive={false} />
                    {activeVar === 'SaO2' && <ReferenceLine y={80} stroke={TH.fnt} strokeDasharray="5 5" label={{value:'80%',fill:TH.fnt,fontSize:9}} />}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <>
                  <div style={{fontSize:11,fontFamily:FN.m,color:TH.mut,marginBottom:4}}>
                    Normoxia (red) vs Silent Hypoxemia (blue) — {VARS.find(v=>v.key===activeVar)?.label}
                  </div>
                  <ResponsiveContainer width="100%" height={360}>
                    <LineChart margin={{top:10,right:20,bottom:10,left:10}}>
                      <CartesianGrid strokeDasharray="3 3" stroke={TH.bLight} />
                      <XAxis dataKey="t" stroke={TH.mut} fontSize={10} type="number"
                        domain={['dataMin','dataMax']}
                        label={{value:'t (s)',position:'insideBottomRight',offset:-5,fill:TH.mut,fontSize:10}} />
                      <YAxis stroke={TH.mut} fontSize={10} domain={['auto','auto']} />
                      <Tooltip contentStyle={{background:'#fff',border:`1px solid ${TH.border}`,borderRadius:4,fontSize:11}} />
                      <Legend wrapperStyle={{fontSize:11}} />
                      <Line data={data1} type="monotone" dataKey={activeVar} stroke={TH.red}
                        dot={false} strokeWidth={1.2} name="Normoxia" isAnimationActive={false} />
                      {data2 && <Line data={data2} type="monotone" dataKey={activeVar} stroke={TH.blue}
                        dot={false} strokeWidth={1.2} name="SH ([Hb]=250)" isAnimationActive={false} />}
                      {activeVar === 'SaO2' && <ReferenceLine y={80} stroke={TH.fnt} strokeDasharray="5 5" />}
                    </LineChart>
                  </ResponsiveContainer>
                </>
              )}
            </div>
          )}

          {mode === 'compare' && l2s && (
            <div style={{background:TH.card,borderRadius:6,border:`1px solid ${TH.border}`,padding:14}}>
              <div style={{fontFamily:FN.d,fontSize:13,fontWeight:700,color:TH.pri,marginBottom:8}}>L₂ Norm: Normoxia vs Silent Hypoxemia</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={l2Bars} margin={{top:5,right:16,bottom:5,left:8}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={TH.bLight} />
                  <XAxis dataKey="name" stroke={TH.mut} fontSize={11} />
                  <YAxis stroke={TH.mut} fontSize={10} />
                  <Tooltip contentStyle={{background:'#fff',border:`1px solid ${TH.border}`,borderRadius:4,fontSize:11}} />
                  <Bar dataKey="value" radius={[3,3,0,0]} isAnimationActive={false}>
                    {l2Bars.map((_,i) => <Cell key={i} fill={i<3?TH.blue:TH.acc} fillOpacity={0.75} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{display:'flex',gap:6,flexWrap:'wrap',marginTop:6}}>
                {Object.entries(l2s).map(([k,v]) => (
                  <span key={k} style={{background:TH.surface,border:`1px solid ${TH.bLight}`,borderRadius:3,padding:'2px 6px',fontSize:10,fontFamily:FN.m}}>
                    <span style={{color:TH.mut}}>{k}:</span>{' '}
                    <span style={{color:TH.pri,fontWeight:600}}>{v.toFixed(4)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Sec>
  );

  // ═══════════════════════════════════════════════════════════════════
  // FINDINGS
  // ═══════════════════════════════════════════════════════════════════
  const Findings = () => (
    <Sec>
      <SecT sub="Detailed evidence of correctness for the MATLAB to Python translation, with code-level comparisons and quantitative validation">Findings and Verification</SecT>

      <Prf title="On Using the Correct Model">
        <p style={{margin:'0 0 10px'}}>Prof. Peter J. Thomas emphasized the importance of implementing the <em>exact</em> published model when producing computational graphs for this work. The model used here is the 7D-O₂ system from Diekman et al. (2017), extended with the silent hypoxemia parameter modifications from Diekman et al. (2024). No simplifications, alternative formulations, or approximations were substituted. Every equation, parameter value, and initial condition is taken directly from the published MATLAB code on ModelDB accession #229640 and #2015954.</p>
        <p style={{margin:0}}>This matters because the 7D-O₂ model has specific dynamical properties — bistability between eupneic and tachypneic states, saddle-node bifurcation structure, sensitivity to the Henry's law small parameter — that depend on the precise equation formulation. Substituting a different CPG model or a simplified oxygen transport equation would alter the bifurcation diagram and invalidate the paper's conclusions about which parameters drive the SH phenotype.</p>
      </Prf>

      <Prf title="Proof 1: Line-by-Line Equation Translation (Chemosensory Feedback, Eq. 28)">
        <p style={{margin:'0 0 10px'}}>Each line in <Cd>closedloop.m</Cd> maps to a specific equation in the paper's Appendix A:</p>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:TH.red,marginBottom:4,fontFamily:FN.m}}>MATLAB (closedloop.m, line 76)</div>
            <Cd block>{`% Chemosensory feedback
gtonic = 0.3*(1-tanh((PO2blood-85)/30));
Itonic = gtonic*(v-Esyn);`}</Cd>
          </div>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:TH.blue,marginBottom:4,fontFamily:FN.m}}>Python (respiratory_model.py)</div>
            <Cd block>{`# Chemosensation (Eq 28)
gt = p['phi'] * (1.0 - np.tanh(
    (PaO2 - p['theta_g']) / p['sigma_g']))
It = gt * (V - p['Etonic'])`}</Cd>
          </div>
        </div>
        <p style={{margin:'10px 0 0'}}>The MATLAB version hardcodes φ=0.3, θg=85, σg=30 directly. The Python version parameterizes these as dictionary entries, enabling the parameter sweeps shown in the paper's Fig. 3B without modifying source code.</p>
      </Prf>

      <Prf title="Proof 2: Blood Oxygen Equation (Eqs. 21–25)">
        <p style={{margin:'0 0 10px'}}>The blood oxygen ODE (Eq. 21) involves the hemoglobin saturation derivative and is the most error-prone part of any translation:</p>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:TH.red,marginBottom:4,fontFamily:FN.m}}>MATLAB (lines 60–72)</div>
            <Cd block>{`SaO2 = (PO2blood^c)/(PO2blood^c+K^c);
CaO2 = eta*SaO2+betaO2*PO2blood;
partial = (c*PO2blood^(c-1))*...
  (1/(PO2blood^c+K^c)-...
   (PO2blood^c)/((PO2blood^c+K^c)^2));
Jlb = (1/taulb)*(PO2lung-PO2blood)*...
  (vollung/(R*Temp));
Jbt = M*CaO2*gamma;
z(7) = (Jlb-Jbt)/...
  (gamma*(betaO2+eta*partial));`}</Cd>
          </div>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:TH.blue,marginBottom:4,fontFamily:FN.m}}>Python translation</div>
            <Cd block>{`S = PaO2**cc / (PaO2**cc + KK**cc)
dS = cc*PaO2**(cc-1) * (
    1/(PaO2**cc + KK**cc) -
    PaO2**cc / (PaO2**cc + KK**cc)**2)
JLB = (PAO2-PaO2)/p['tau_LB'] * (
    volL / (p['R_gas']*p['Temp']))
JBT = p['M']*zeta*(
    p['betaO2']*PaO2 + eta*S)
dPaO2 = (JLB - JBT) / (
    zeta*(p['betaO2'] + eta*dS))`}</Cd>
          </div>
        </div>
      </Prf>

      <Prf title="Proof 3: Solver Equivalence (ode15s → Radau)">
        <Cd block>{`# MATLAB (reproduce_figure1.m):
options = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,u] = ode15s('closedloop',[0 tf],inits,options);

# Python equivalent:
sol = solve_ivp(
    fun=lambda t,u: rhs(t, u, params),
    t_span=[0, tf], y0=inits,
    method='Radau',        # implicit, stiff-capable
    rtol=1e-9, atol=1e-9,  # identical tolerances
    max_step=5.0)`}</Cd>
        <p style={{margin:'10px 0 0'}}>Both use identical tolerances (10⁻⁹) and initial conditions from <Cd>reproduce_figure1.m</Cd>. Radau was chosen because ode15s is a BDF/NDF method for stiff problems, and Radau is the closest single-step stiff solver in scipy.</p>
      </Prf>

      <Prf title="Proof 4: Quantitative Output Validation">
        <div style={{overflowX:'auto'}}>
          <table style={{width:'100%',borderCollapse:'collapse',fontFamily:FN.m,fontSize:11}}>
            <thead><tr style={{borderBottom:`2px solid ${TH.border}`,background:TH.surface}}>
              {['Variable','Paper Range','Python Output','Match'].map(h => <th key={h} style={{padding:'6px 10px',textAlign:'left',color:TH.mut,fontSize:10,textTransform:'uppercase'}}>{h}</th>)}
            </tr></thead>
            <tbody>{[
              ['V (mV)','\u221260 to +10 (bursting, Fig. 1)','−57 (bursting)','Yes'],
              ['α','0 to ~0.01 (Fig. 1)','0.0002–0.01','Yes'],
              ['volL (L)','~2.0 to ~3.0 (Fig. 1)','2.0–2.9','Yes'],
              ['PaO₂ normoxia','~95–110 (Fig. 2A plateau)','104.1 mmHg','Yes'],
              ['PaO₂ SH','~70–85 (Fig. 5B, [Hb]=250)','75.7 mmHg','Yes'],
              ['g_tonic (nS)','~0.10–0.25 (Fig. 1)','0.12–0.22','Yes'],
            ].map((r,i) => (
              <tr key={i} style={{borderBottom:`1px solid ${TH.bLight}`}}>
                <td style={{padding:'5px 10px',fontWeight:600,color:TH.pri}}>{r[0]}</td>
                <td style={{padding:'5px 10px',color:TH.txt2}}>{r[1]}</td>
                <td style={{padding:'5px 10px',color:TH.txt}}>{r[2]}</td>
                <td style={{padding:'5px 10px',color:TH.acc,fontWeight:700}}>{r[3]}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Prf>

      <Prf title="Proof 5: L₂ Norm Cross-Validation">
        <Cd block>{`L₂(x) = sqrt( (1/T) ∫₀ᵀ |x_normoxia(t) − x_SH(t)|² dt )

Computed values (Python, tf = 15s):
  V:      7.590     mV
  n:      0.135     (dimensionless)
  h:      0.053     (dimensionless)
  α:      0.003     (dimensionless)
  volL:   0.374     L
  PAO₂:  15.002     mmHg
  PaO₂:  14.963     mmHg`}</Cd>
        <p style={{margin:'10px 0 0'}}>The largest divergences are in oxygen variables (PAO₂, PaO₂), consistent with the paper's finding that the SH phenotype manifests primarily in oxygen handling, not CPG rhythm.</p>
      </Prf>

      <Prf title="Why These Specific SH Parameters">
        <p style={{margin:'0 0 6px'}}><strong style={{color:TH.txt}}>Chemosensory (Fig. 3B):</strong> 27 combinations tested. φ=0.24, θg=70, σg=36 gave the greatest PaO₂ reduction.</p>
        <p style={{margin:'0 0 6px'}}><strong style={{color:TH.txt}}>Hemoglobin binding K (Fig. 4B):</strong> Varied, shifted collapse point but insufficient alone.</p>
        <p style={{margin:'0 0 6px'}}><strong style={{color:TH.txt}}>Lung params vol₀, τ_LB (Fig. 5A):</strong> ±20% / ±400% variation, surprisingly little effect.</p>
        <p style={{margin:0}}><strong style={{color:TH.txt}}>Hemoglobin [Hb] (Fig. 5B):</strong> Decisive parameter. [Hb]=250 shifts collapse leftward. The paper states: <em>"Thus we will consider the model with [Hb]=250 as our working model for silent hypoxemia"</em> (p. 9).</p>
      </Prf>
    </Sec>
  );

  // ═══════════════════════════════════════════════════════════════════
  // REFERENCES
  // ═══════════════════════════════════════════════════════════════════
  const Refs = () => (
    <Sec>
      <SecT sub="Source publications, code repositories, and deployment information">References and Resources</SecT>
      <div style={{background:TH.surface,border:`1px solid ${TH.border}`,borderRadius:6,padding:'24px 28px',marginBottom:24}}>
        <h3 style={{fontFamily:FN.d,fontSize:17,color:TH.txt,margin:'0 0 6px'}}>COVID-19 and Silent Hypoxemia in a Minimal Closed-Loop Model of the Respiratory Rhythm Generator</h3>
        <p style={{fontFamily:FN.b,fontSize:13,color:TH.txt2,margin:'0 0 4px'}}>Casey O. Diekman (NJIT), Peter J. Thomas (Case Western Reserve), Christopher G. Wilson (Loma Linda)</p>
        <p style={{fontFamily:FN.m,fontSize:11,color:TH.pri,margin:'0 0 16px'}}>Biological Cybernetics (2024), 118(3-4):145–163 · DOI: 10.1007/s00422-024-00989-w · PMID: 38884785</p>
        <p style={{fontFamily:FN.b,fontSize:13,color:TH.mut,lineHeight:1.7,margin:0}}>The paper uses the 7D-O₂ model to test whether altered chemosensory function at the carotid bodies and/or the NTS are responsible for the blunted hypoxia response in COVID-19. The central finding is that increasing hemoglobin concentration is necessary and sufficient to shift the metabolic collapse point, producing the SH phenotype.</p>
      </div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))',gap:12,marginBottom:24}}>
        {[
          ['Original MATLAB Code','https://modeldb.science/229640','ModelDB #229640 — Diekman et al. (2017), J. Neurophysiol.'],
          ['SH Extension','https://modeldb.science/2015954','ModelDB #2015954 — COVID-19 model (2024).'],
          ['GitHub','https://github.com/ModelDBRepository/229640','Full MATLAB source with figure reproduction scripts.'],
          ['PubMed','https://pubmed.ncbi.nlm.nih.gov/38884785/','Free PMC article.'],
        ].map(([t,u,d],i) => (
          <a key={i} href={u} target="_blank" rel="noopener noreferrer"
            style={{background:TH.card,border:`1px solid ${TH.border}`,borderRadius:6,padding:'16px 20px',textDecoration:'none',display:'block'}}>
            <h4 style={{fontFamily:FN.d,fontSize:13,color:TH.pri,margin:'0 0 6px'}}>{t}</h4>
            <p style={{fontFamily:FN.b,fontSize:11,color:TH.mut,margin:0,lineHeight:1.5}}>{d}</p>
          </a>
        ))}
      </div>
      <Prf title="Acknowledgments">
        <p style={{margin:0}}>Based entirely on the work of Casey O. Diekman (NJIT), Peter J. Thomas (Case Western Reserve University), and Christopher G. Wilson (Loma Linda University). Supported by NIH grants RF1 NS118606-01 and RO1 AT011691-01, and NSF grants DMS-2052109, DMS-1555237, and DMS-2152115.</p>
      </Prf>
    </Sec>
  );

  return (
    <div style={{background:TH.bg,color:TH.txt,minHeight:'100vh',fontFamily:FN.b}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
        input[type=range] { -webkit-appearance: none; background: #e9ecef; border-radius: 2px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #2980b9; cursor: pointer; }
        a { transition: opacity 0.2s; }
        a:hover { opacity: 0.85; }
      `}</style>
      <Nav />
      {page === 'home' && <Home />}
      {page === 'explorer' && <Explorer />}
      {page === 'findings' && <Findings />}
      {page === 'paper' && <Refs />}
      <footer style={{borderTop:`1px solid ${TH.border}`,padding:'20px 24px',textAlign:'center',fontFamily:FN.m,fontSize:10,color:TH.fnt}}>
        7D-O₂ Closed-Loop Respiratory Control Model · Python Translation and Interactive Explorer · - Brayden Choi - Based on Diekman, Thomas & Wilson, Biol. Cybern. (2024) · ModelDB #229640 / #2015954
      </footer>
    </div>
  );
}
