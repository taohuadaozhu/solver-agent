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

/* ── AgentPanel — LLM-driven multi-tool agent with SSE streaming ───────── */

function AgentPanel({ query }) {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [finalAnswer, setFinalAnswer] = useState('');
  const [charts, setCharts] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_WORKFLOW}/workflow/agent-stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          let eventType = '';
          let eventData = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6).trim();
            } else if (line === '' && eventType) {
              try {
                const data = JSON.parse(eventData);
                if (cancelled) return;
                switch (eventType) {
                  case 'thinking':
                    setEvents((prev) => [...prev, { type: 'thinking', round: data.round }]);
                    break;
                  case 'tool_calls':
                    setEvents((prev) => [...prev, { type: 'tool_calls', calls: data.calls }]);
                    break;
                  case 'tool_result':
                    setEvents((prev) => [...prev, { type: 'tool_result', ...data }]);
                    break;
                  case 'llm_response':
                    setFinalAnswer(data.content);
                    break;
                  case 'done':
                    if (data.charts?.length) setCharts(data.charts);
                    setLoading(false);
                    break;
                  case 'error':
                    setError(data.error);
                    setLoading(false);
                    break;
                }
              } catch { /* skip malformed SSE */ }
              eventType = '';
              eventData = '';
            }
          }
        }
      } catch (e) {
        if (!cancelled) { setError(e.message); setLoading(false); }
      }
    })();
    return () => { cancelled = true; };
  }, [query]);

  const toolLabel = (name) => {
    switch (name) {
      case 'load_dataset': return 'Load Dataset';
      case 'execute_solver': return 'Execute Solver';
      case 'generate_chart': return 'Generate Chart';
      default: return name;
    }
  };

  const toolIcon = (name) => {
    switch (name) {
      case 'load_dataset': return '📂';
      case 'execute_solver': return '⚙️';
      case 'generate_chart': return '📊';
      default: return '🔧';
    }
  };

  return (
    <div className="agent-panel">
      {error && <div className="answer-error">{error}</div>}

      <div className="agent-timeline">
        {events.map((ev, i) => {
          if (ev.type === 'thinking') {
            return (
              <div key={i} className="agent-event thinking-event">
                <span className="agent-event-icon">💭</span>
                <span className="agent-event-text">Round {ev.round} — thinking...</span>
              </div>
            );
          }
          if (ev.type === 'tool_calls') {
            return (
              <div key={i} className="agent-event tool-calls-event">
                {ev.calls.map((tc, j) => (
                  <details key={tc.id || j} className="tool-call-card">
                    <summary className="tool-call-summary">
                      <span>{toolIcon(tc.name)}</span>
                      <span className="tool-call-name">{toolLabel(tc.name)}</span>
                      {tc.arguments?.algorithm && (
                        <span className="tool-call-algo">{tc.arguments.algorithm}</span>
                      )}
                      {tc.arguments?.dataset_name && (
                        <span className="tool-call-ds">{tc.arguments.dataset_name}</span>
                      )}
                    </summary>
                    <pre className="tool-call-args">{JSON.stringify(tc.arguments, null, 2)}</pre>
                  </details>
                ))}
              </div>
            );
          }
          if (ev.type === 'tool_result') {
            const res = ev.result || {};
            const ok = res.success !== false;
            return (
              <div key={i} className="agent-event tool-result-event">
                <details className={`tool-result-card ${ok ? 'result-ok' : 'result-err'}`}>
                  <summary className="tool-result-summary">
                    <span>{ok ? '✅' : '❌'}</span>
                    <span className="tool-result-label">{toolLabel(ev.name)} complete</span>
                    {res.result?.best_objective != null && (
                      <span className="tool-result-metric">Obj: {res.result.best_objective}</span>
                    )}
                    {res.result?.runtime_seconds != null && (
                      <span className="tool-result-metric">{res.result.runtime_seconds}s</span>
                    )}
                  </summary>
                  <pre className="tool-result-data">{JSON.stringify(res, null, 2)}</pre>
                </details>
              </div>
            );
          }
          return null;
        })}
      </div>

      {loading && !error && (
        <div className="loading-dots"><span /><span /><span /></div>
      )}

      {finalAnswer && (
        <div className="agent-final-answer">
          <div className="answer-text">{finalAnswer}</div>
        </div>
      )}

      {charts.map((c, i) => <ChartView key={i} chart={c} />)}
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
  const [skipSteps, setSkipSteps] = useState([]);
  const stateRef = useRef(null);
  const dsRef = useRef('');

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { dsRef.current = selectedDataset; }, [selectedDataset]);

  // Auto-run STEP_1 on mount via /workflow/start (returns coverage info)
  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_WORKFLOW}/workflow/start`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        setState(data.state);
        setOutput((prev) => ({ ...prev, STEP_1_PROBLEM: data.analysis }));
        setCompleted((prev) => new Set([...prev, 'STEP_1_PROBLEM']));
        if (data.datasets?.length > 0) {
          setDatasets(data.datasets);
          setSelectedDataset(data.datasets[0].name);
        }
        // Handle step skipping: auto-fill pre-covered steps
        const skips = data.skip_steps || [];
        setSkipSteps(skips);
        if (skips.length > 0) {
          const newOutput = { STEP_1_PROBLEM: data.analysis };
          const newCompleted = new Set(['STEP_1_PROBLEM']);
          const skipLabels = {
            STEP_3_VARIABLE: '(Pre-filled: variables already described in query)',
            STEP_4_OBJECTIVE: '(Pre-filled: objectives already described in query)',
            STEP_5_CONSTRAINT: '(Pre-filled: constraints already described in query)',
          };
          for (const s of skips) {
            newOutput[s] = skipLabels[s] || '(Auto-filled)';
            newCompleted.add(s);
          }
          setOutput(newOutput);
          setCompleted(newCompleted);
        }
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    })();
  }, []);

  const runStep = async (step, extraInput) => {
    setLoading(true);
    setError('');
    try {
      const body = { step, query };
      if (stateRef.current) body.state_json = JSON.stringify(stateRef.current);
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
      setState((prev) => ({ ...prev, dataset_name: ds }));
      setOutput((prev) => ({ ...prev, STEP_2_DATASET: `Selected: ${ds}` }));
      setCompleted((prev) => new Set([...prev, 'STEP_2_DATASET']));
      setLoading(true);

      const merged = JSON.stringify({ ...(stateRef.current || {}), dataset_name: ds });
      const body = JSON.stringify({ query, selected_dataset: ds, state_json: merged, skip_steps: skipSteps });

      // SSE streaming via fetch + ReadableStream
      (async () => {
        try {
          const res = await fetch(`${API_WORKFLOW}/workflow/stream`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // keep incomplete line in buffer

            let eventType = '';
            let eventData = '';
            for (const line of lines) {
              if (line.startsWith('event: ')) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith('data: ')) {
                eventData = line.slice(6).trim();
              } else if (line === '' && eventType) {
                // End of event — process it
                try {
                  const data = JSON.parse(eventData);
                  _handleSSE(eventType, data);
                } catch { /* skip malformed */ }
                eventType = '';
                eventData = '';
              }
            }
          }
        } catch (e) { setError(e.message); }
        finally { setLoading(false); }
      })();

      // SSE event handler (closure over component state)
      const _handleSSE = (type, data) => {
        switch (type) {
          case 'step_start': {
            const stepMap = { 'STEP_3_VARIABLE': 'STEP_3_VARIABLE', 'STEP_4_OBJECTIVE': 'STEP_4_OBJECTIVE',
              'STEP_5_CONSTRAINT': 'STEP_5_CONSTRAINT', 'STEP_6_CLASSIFY': 'STEP_6_CLASSIFY', 'STEP_7_ALGO': 'STEP_7_ALGO' };
            const s = stepMap[data.step] || data.step;
            setCurrentStep(s);
            // Initialize streaming output buffer
            setOutput((prev) => ({ ...prev, [s]: '' }));
            break;
          }
          case 'step_chunk': {
            // Append streaming token to current step output
            setOutput((prev) => ({ ...prev, [data.step]: (prev[data.step] || '') + data.text }));
            break;
          }
          case 'step_done': {
            if (!data.skipped) {
              setOutput((prev) => ({ ...prev, [data.step]: data.output }));
            }
            setCompleted((prev) => new Set([...prev, data.step]));
            setState((prev) => ({ ...(prev || {}), step_results: { ...(prev?.step_results || {}), [data.step]: data.output } }));
            break;
          }
          case 'done': {
            setState(data.state);
            setCompleted((prev) => new Set([...prev, 'STEP_7_ALGO']));
            setCurrentStep('STEP_7_ALGO');
            break;
          }
          case 'error': {
            setError(data.error);
            break;
          }
        }
      };
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
      setExecResult({ ...data.execution, validation: data.validation });
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
            <div className="metric"><span className="metric-value">{execResult.status}</span><span className="metric-label">Status</span></div>
          </div>
          {execResult.validation && (
            <div className="validation-badges">
              {Object.entries(execResult.validation).filter(([k]) => k !== 'all_pass').map(([key, ok]) => (
                <span key={key} className={`val-badge ${ok ? 'val-pass' : 'val-fail'}`}>
                  {ok ? '✓' : '✗'} {key}
                </span>
              ))}
            </div>
          )}
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
        {msg.agentQuery && <AgentPanel query={msg.agentQuery} />}
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

  const handleSendAgent = (query) => {
    saveMsg('user', query);
    setMessages((prev) => [...prev, { role: 'user', content: query }, { role: 'assistant', agentQuery: query }]);
    setInput('');
  };

  const handleSend = () => {
    const query = input.trim();
    if (!query || loading) return;
    if (mode === 'agent') handleSendAgent(query);
    else if (mode === 'workflow') handleSendWorkflow(query);
    else handleSendQuick(query);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div className="chat-page">
      <div className="chat-controls">
        <div className="mode-switch">
          <button className={mode === 'workflow' ? 'mode-active' : ''} onClick={() => setMode('workflow')}>Guided Workflow</button>
          <button className={mode === 'agent' ? 'mode-active' : ''} onClick={() => setMode('agent')}>Agent Mode</button>
          <button className={mode === 'quick' ? 'mode-active' : ''} onClick={() => setMode('quick')}>Quick Recommend</button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>Solver Agent</h2>
            <p>{mode === 'workflow'
              ? 'Describe your optimization problem. The agent will guide you step by step: problem → dataset → variables → objectives → constraints → classify → algorithm → execute.'
              : mode === 'agent'
              ? 'Describe your optimization problem. The LLM agent will autonomously load datasets, execute solvers, generate charts, and compare results.'
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
