import { useState, useRef, useEffect, useCallback } from 'react';

const API_RECOMMEND = 'http://localhost:8000';
const API_WORKFLOW = 'http://localhost:8001';

const STEPS = [
  { key: 'STEP_1_PROBLEM', label: 'Problem', num: 1 },
  { key: 'STEP_2_DATASET', label: 'Dataset', num: 2 },
  { key: 'STEP_3_VARIABLE', label: 'Variables', num: 3 },
  { key: 'STEP_4_OBJECTIVE', label: 'Objectives', num: 4 },
  { key: 'STEP_5_CONSTRAINT', label: 'Constraints', num: 5 },
  { key: 'STEP_6_CLASSIFY', label: 'Classify', num: 6 },
  { key: 'STEP_7_ALGO', label: 'Algorithm', num: 7 },
];

const AUTO_STEPS = new Set(['STEP_1_PROBLEM', 'STEP_3_VARIABLE', 'STEP_6_CLASSIFY', 'STEP_7_ALGO']);

function StepBar({ current, completed }) {
  return (
    <div className="step-bar">
      {STEPS.map((s, i) => {
        const done = completed.has(s.key);
        const active = s.key === current;
        return (
          <div key={s.key} className={`step-dot-wrap ${i < STEPS.length - 1 ? 'step-dot-line' : ''}`}>
            <div className={`step-dot ${done ? 'dot-done' : ''} ${active ? 'dot-active' : ''}`}>
              {done ? '✓' : s.num}
            </div>
            <span className={`step-dot-label ${active ? 'label-active' : ''}`}>{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}

function DocBadge({ doc, index }) {
  return (
    <div className="doc-badge">
      <span className="doc-rank">#{index + 1}</span>
      <div className="doc-info">
        <span className="doc-name">{doc.name}</span>
        <span className="doc-sim">{(doc.similarity * 100).toFixed(0)}%</span>
      </div>
      <p className="doc-summary">{doc.summary}</p>
    </div>
  );
}

function ChartView({ chart }) {
  if (!chart || !chart.data) return null;
  const id = `chart-${Math.random().toString(36).slice(2)}`;

  useEffect(() => {
    if (window.Plotly) {
      window.Plotly.newPlot(id, chart.data, chart.layout, {
        responsive: true,
        displayModeBar: false,
        margin: { t: 40, r: 20, b: 40, l: 40 },
      });
    }
  }, [id, chart]);

  return <div id={id} className="chart-container" />;
}

/* ── WorkflowPanel — drives the step-by-step interactive workflow ───── */

function WorkflowPanel({ query, onDone, saveWorkflow }) {
  const [currentStep, setCurrentStep] = useState('STEP_1_PROBLEM');
  const [completed, setCompleted] = useState(new Set());
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState({});
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [userInput, setUserInput] = useState('');
  const [execResult, setExecResult] = useState(null);
  const [charts, setCharts] = useState([]);
  const [error, setError] = useState('');
  const stateRef = useRef(null);
  const dsRef = useRef('');

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { dsRef.current = selectedDataset; }, [selectedDataset]);

  // Auto-run STEP_1 on mount
  useEffect(() => {
    runStep('STEP_1_PROBLEM', query);
  }, []);

  const runStep = async (step, extraInput) => {
    setLoading(true);
    setError('');
    try {
      const body = { step, query };
      if (state) body.state_json = JSON.stringify(state);
      if (extraInput) body.query = extraInput;

      const res = await fetch(`${API_WORKFLOW}/workflow/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setState(data.state);
      setOutput((prev) => ({ ...prev, [step]: data.response }));
      setCompleted((prev) => new Set([...prev, step]));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // After each step completes, handle special logic
  useEffect(() => {
    const stateObj = state || {};
    if (currentStep === 'STEP_1_PROBLEM' && completed.has('STEP_1_PROBLEM')) {
      // Fetch dataset options via RAG
      fetchDatasets();
    }
    if (currentStep === 'STEP_7_ALGO' && completed.has('STEP_7_ALGO')) {
      // Workflow complete
    }
  }, [completed, currentStep]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_RECOMMEND}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 10, type: 'datasets' }),
      });
      if (res.ok) {
        const data = await res.json();
        setDatasets(data.results || []);
        if (data.results?.[0]) setSelectedDataset(data.results[0].name);
      }
    } catch { /* ignore */ }
  };

  const getStateJson = (overrides) => JSON.stringify({ ...(stateRef.current || {}), ...(overrides || {}) });

  const handleNext = () => {
    // Dataset picker confirmation (currentStep is still STEP_1, but dataset picker is showing)
    if (showDatasetPicker) {
      const ds = dsRef.current;
      if (!ds) return;
      const nextStep = 'STEP_3_VARIABLE';
      setState((prev) => ({ ...prev, dataset_name: ds }));
      setOutput((prev) => ({ ...prev, STEP_2_DATASET: `Selected: ${ds}` }));
      setCompleted((prev) => new Set([...prev, 'STEP_2_DATASET']));
      setCurrentStep(nextStep);
      setLoading(true);
      const merged = JSON.stringify({ ...(stateRef.current || {}), dataset_name: ds });
      (async () => {
        try {
          const res = await fetch(`${API_WORKFLOW}/workflow/step`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ step: nextStep, query, state_json: merged }),
          });
          if (!res.ok) throw new Error('Failed');
          const data = await res.json();
          setState(data.state);
          setOutput((prev) => ({ ...prev, [nextStep]: data.response }));
          setCompleted((prev) => new Set([...prev, nextStep]));
        } catch (e) { setError(e.message); }
        finally { setLoading(false); }
      })();
      return;
    }

    const idx = STEPS.findIndex((s) => s.key === currentStep);
    if (idx < STEPS.length - 1) {
      const nextStep = STEPS[idx + 1].key;

      if (currentStep === 'STEP_3_VARIABLE') {
        setCurrentStep(nextStep);
        return;
      }

      if (currentStep === 'STEP_4_OBJECTIVE') {
        if (!userInput.trim()) return;
        setOutput((prev) => ({ ...prev, STEP_4_OBJECTIVE: userInput }));
        setCurrentStep(nextStep);
        setLoading(true);
        (async () => {
          try {
            const res = await fetch(`${API_WORKFLOW}/workflow/step`, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ step: nextStep, query: `Objectives: ${userInput}`, state_json: getStateJson() }),
            });
            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            setState(data.state);
            setOutput((prev) => ({ ...prev, [nextStep]: data.response }));
            setCompleted((prev) => new Set([...prev, nextStep]));
            setUserInput('');
          } catch (e) { setError(e.message); }
          finally { setLoading(false); }
        })();
        return;
      }

      if (currentStep === 'STEP_5_CONSTRAINT') {
        if (!userInput.trim()) return;
        setOutput((prev) => ({ ...prev, STEP_5_CONSTRAINT: userInput }));
        setCurrentStep(nextStep);
        setLoading(true);
        (async () => {
          try {
            const res = await fetch(`${API_WORKFLOW}/workflow/step`, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ step: nextStep, query: `Constraints: ${userInput}`, state_json: getStateJson() }),
            });
            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            setState(data.state);
            setOutput((prev) => ({ ...prev, [nextStep]: data.response }));
            setCompleted((prev) => new Set([...prev, nextStep]));
            setUserInput('');
          } catch (e) { setError(e.message); }
          finally { setLoading(false); }
        })();
        return;
      }

      if (currentStep === 'STEP_6_CLASSIFY') {
        setCurrentStep(nextStep);
        setLoading(true);
        (async () => {
          try {
            const res = await fetch(`${API_WORKFLOW}/workflow/step`, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ step: nextStep, query, state_json: getStateJson() }),
            });
            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            setState(data.state);
            setOutput((prev) => ({ ...prev, [nextStep]: data.response }));
            setCompleted((prev) => new Set([...prev, nextStep]));
          } catch (e) { setError(e.message); }
          finally { setLoading(false); }
        })();
        return;
      }
    }
  };

  const handleExecute = async () => {
    const algo = state?.recommended_algorithm;
    const ds = state?.dataset_name || selectedDataset;
    if (!algo) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_WORKFLOW}/workflow/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: algo, dataset_name: ds, parameters: { max_iterations: 200, population_size: 50 } }),
      });
      if (!res.ok) throw new Error('Execute failed');
      const data = await res.json();
      setExecResult(data.execution);
      if (data.chart) setCharts([data.chart]);

      // Auto-save completed workflow
      const updatedState = { ...(stateRef.current || {}), execution_result: data.execution };
      if (saveWorkflow) {
        saveWorkflow(query, updatedState, output);
      }
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const step1Done = completed.has('STEP_1_PROBLEM');
  const showDatasetPicker = step1Done && datasets.length > 0 && !completed.has('STEP_2_DATASET');
  const showStepOutput = currentStep !== 'STEP_1_PROBLEM' || !showDatasetPicker;

  const isLast = completed.has('STEP_7_ALGO');
  const needsUserInput = currentStep === 'STEP_4_OBJECTIVE' || currentStep === 'STEP_5_CONSTRAINT';

  // Effective displayed step for the progress bar
  const displayStep = showDatasetPicker ? 'STEP_2_DATASET' : currentStep;

  const algo = state?.recommended_algorithm || '';

  return (
    <div className="workflow-panel">
      <StepBar current={displayStep} completed={completed} />

      {error && <div className="answer-error">{error}</div>}

      {/* Step output (auto-steps) */}
      {showStepOutput && (
        <div className="step-output">
          {output[currentStep] && (
            <pre className="step-text">{output[currentStep]}</pre>
          )}
          {loading && <div className="loading-dots"><span /><span /><span /></div>}
        </div>
      )}

      {/* Dataset picker — shows right after STEP_1, no button needed */}
      {showDatasetPicker && (
        <div className="dataset-picker">
          <h4>Select a dataset for your problem:</h4>
          <div className="dataset-list">
            {datasets.map((ds, i) => (
              <label key={i} className={`dataset-option ${selectedDataset === ds.name ? 'dataset-selected' : ''}`}>
                <input type="radio" name="ds" checked={selectedDataset === ds.name} onChange={() => setSelectedDataset(ds.name)} />
                <div>
                  <span className="ds-name">{ds.name}</span>
                  <span className="ds-sim">{(ds.similarity * 100).toFixed(0)}% match</span>
                </div>
                <p>{ds.summary}</p>
              </label>
            ))}
          </div>
          <button
            className="next-btn"
            style={{ marginTop: 8 }}
            onClick={handleNext}
            disabled={!selectedDataset || loading}
          >
            {loading ? 'Loading...' : 'Confirm Dataset → Variables'}
          </button>
        </div>
      )}

      {/* User input for objectives/constraints */}
      {needsUserInput && (
        <div className="user-input-area">
          <textarea
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={currentStep === 'STEP_4_OBJECTIVE'
              ? 'e.g. Minimize total completion time, minimize cost...'
              : 'e.g. Each machine can only process one job at a time, precedence constraints...'}
            rows={4}
            disabled={loading}
          />
        </div>
      )}

      {/* Execution result */}
      {execResult && (
        <div className="execution-result">
          <h3>Execution Result</h3>
          <div className="exec-metrics">
            <div className="metric"><span className="metric-value">{execResult.best_objective}</span><span className="metric-label">Best Objective</span></div>
            <div className="metric"><span className="metric-value">{execResult.runtime_seconds}s</span><span className="metric-label">Runtime</span></div>
            <div className="metric"><span className="metric-value">{execResult.iterations}</span><span className="metric-label">Iterations</span></div>
            <div className="metric"><span className={`metric-value quality-${execResult.solution_quality}`}>{execResult.solution_quality}</span><span className="metric-label">Quality</span></div>
          </div>
          <pre className="exec-output">{execResult.raw_output}</pre>
        </div>
      )}

      {charts.map((c, i) => <ChartView key={i} chart={c} />)}

      {/* Action buttons */}
      <div className="step-actions">
        {currentStep === 'STEP_3_VARIABLE' && completed.has('STEP_3_VARIABLE') && (
          <button className="next-btn" onClick={handleNext} disabled={loading}>
            {loading ? 'Processing...' : 'Next: Input Objectives'}
          </button>
        )}
        {needsUserInput && (
          <button className="next-btn" onClick={handleNext} disabled={!userInput.trim() || loading}>
            {loading ? 'Processing...' : currentStep === 'STEP_4_OBJECTIVE' ? 'Next: Input Constraints' : 'Next: Classify Problem'}
          </button>
        )}
        {currentStep === 'STEP_6_CLASSIFY' && completed.has('STEP_6_CLASSIFY') && !completed.has('STEP_7_ALGO') && (
          <button className="next-btn" onClick={handleNext} disabled={loading}>
            {loading ? 'Processing...' : 'Next: Algorithm Recommendation'}
          </button>
        )}
        {isLast && !execResult && (
          <button className="execute-btn" onClick={handleExecute} disabled={loading}>
            {loading ? 'Executing...' : `Execute ${algo}`}
          </button>
        )}
      </div>
    </div>
  );
}

/* ── Message ───────────────────────────────────────────────────────────── */

function Message({ msg }) {
  return (
    <div className={`message ${msg.role === 'user' ? 'message-user' : 'message-bot'}`}>
      <div className="message-role">{msg.role === 'user' ? 'You' : 'Solver Agent'}</div>
      <div className="message-content">
        {msg.role === 'user' && <div className="user-text">{msg.content}</div>}
        {msg.workflowQuery && <WorkflowPanel query={msg.workflowQuery} saveWorkflow={msg.saveWorkflow} />}
        {msg.answer && <div className="answer-text">{msg.answer}</div>}
        {msg.error && <div className="answer-error">{msg.error}</div>}
      </div>
    </div>
  );
}

/* ── ChatPage ──────────────────────────────────────────────────────────── */

function useMemory() {
  const [convId, setConvId] = useState(() => sessionStorage.getItem('conv_id') || null);

  const ensureConv = useCallback(async () => {
    if (convId) return convId;
    try {
      const res = await fetch(`${API_WORKFLOW}/memory/conversations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: '' }) });
      const data = await res.json();
      const id = data.conversation_id;
      sessionStorage.setItem('conv_id', id);
      setConvId(id);
      return id;
    } catch { return null; }
  }, [convId]);

  const saveMsg = useCallback(async (role, content) => {
    const id = await ensureConv();
    if (!id) return;
    try {
      await fetch(`${API_WORKFLOW}/memory/messages`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: id, role, content }),
      });
    } catch { /* ignore */ }
  }, [ensureConv]);

  const loadHistory = useCallback(async () => {
    const savedId = sessionStorage.getItem('conv_id');
    if (!savedId) return [];
    try {
      const res = await fetch(`${API_WORKFLOW}/memory/conversations/${savedId}/messages`);
      if (!res.ok) return [];
      const data = await res.json();
      setConvId(savedId);
      return data.messages || [];
    } catch { return []; }
  }, []);

  const saveWorkflow = useCallback(async (problem, stateObj, stepOutputs) => {
    const id = await ensureConv();
    if (!id) return;
    try {
      await fetch(`${API_WORKFLOW}/memory/workflows`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: id, problem,
          state_json: JSON.stringify(stateObj || {}),
          step_outputs: stepOutputs || {},
          status: 'completed',
        }),
      });
    } catch { /* ignore */ }
  }, [ensureConv]);

  return { convId, saveMsg, loadHistory, saveWorkflow, ensureConv };
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('workflow');
  const bottomRef = useRef(null);
  const { convId, saveMsg, loadHistory, saveWorkflow } = useMemory();
  const lastWorkflowRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // Load conversation history on mount
  useEffect(() => {
    (async () => {
      const history = await loadHistory();
      if (history.length > 0) {
        const restored = [];
        for (const m of history) {
          if (m.role === 'user') {
            restored.push({ role: 'user', content: m.content });
          } else {
            try {
              const meta = JSON.parse(m.metadata || '{}');
              restored.push({ role: 'assistant', content: m.content, ...meta });
            } catch {
              restored.push({ role: 'assistant', content: m.content });
            }
          }
        }
        setMessages(restored);
      }
    })();
  }, []);

  const handleSendQuick = async (query) => {
    saveMsg('user', query);
    const userMsg = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const res = await fetch(`${API_RECOMMEND}/recommend`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const data = await res.json();
      saveMsg('assistant', data.answer);
      setMessages((prev) => [...prev, { role: 'assistant', content: query, answer: data.answer }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'error', error: err.message }]);
    } finally { setLoading(false); }
  };

  const handleSendWorkflow = (query) => {
    saveMsg('user', query);
    setMessages((prev) => [...prev, { role: 'user', content: query }, { role: 'assistant', workflowQuery: query, saveWorkflow }]);
    setInput('');
  };

  const handleSend = () => {
    const query = input.trim();
    if (!query || loading) return;
    mode === 'workflow' ? handleSendWorkflow(query) : handleSendQuick(query);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div className="chat-page">
      <div className="chat-controls">
        <div className="mode-switch">
          <button className={mode === 'workflow' ? 'mode-active' : ''} onClick={() => setMode('workflow')}>Guided Workflow</button>
          <button className={mode === 'quick' ? 'mode-active' : ''} onClick={() => setMode('quick')}>Quick Recommend</button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>Solver Agent</h2>
            <p>{mode === 'workflow'
              ? 'Describe your optimization problem. The agent will guide you step by step: problem → dataset → variables → objectives → constraints → classify → algorithm → execute.'
              : 'Quick RAG-based algorithm recommendation.'}</p>
            <div className="example-queries">
              <span>Try:</span>
              {['5台机器20个工件最小化完工时间', '多目标生产调度推荐什么算法？', '10辆车50个客户的容量约束路径规划'].map(q => (
                <button key={q} onClick={() => setInput(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <textarea value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown}
          placeholder="Describe your optimization problem..." rows={3} disabled={loading} />
        <button onClick={handleSend} disabled={loading || !input.trim()}>{loading ? '...' : 'Send'}</button>
      </div>
    </div>
  );
}
