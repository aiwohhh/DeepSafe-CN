import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import './App.css';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

const API_BASE_URL = window.location.hostname === 'localhost' && window.location.port === '3000'
  ? 'http://localhost:8000'
  : '/api';

const ProtectedRoute = ({ children, isAuthenticated }) => {
    const location = useLocation();
    if (!isAuthenticated) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }
    return children;
};

function Dashboard({ onLogout }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [mediaType, setMediaType] = useState('image');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [systemHealth, setSystemHealth] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE_URL}/health`)
            .then((res) => res.json())
            .then((data) => setSystemHealth(data))
            .catch(() => setSystemHealth(null));
    }, []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (!file) return;
        setSelectedFile(file);
        setResults(null);
        setError(null);
        setPreviewUrl(URL.createObjectURL(file));
        if (file.type.startsWith('video/')) setMediaType('video');
        else if (file.type.startsWith('audio/')) setMediaType('audio');
        else setMediaType('image');
    };

    const loadDemoFile = async (path, name, type, targetType) => {
        try {
            const res = await fetch(path);
            const blob = await res.blob();
            const file = new File([blob], name, { type });
            setSelectedFile(file);
            setPreviewUrl(path);
            setMediaType(targetType);
            setResults(null);
            setError(null);
        } catch {
            setError('加载演示文件失败');
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;
        setIsAnalyzing(true);
        setError(null);
        setResults(null);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('media_type', mediaType);

        try {
            const response = await fetch(`${API_BASE_URL}/detect`, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '检测失败');
            }
            setResults(await response.json());
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const renderHealthStatus = () => {
        if (!systemHealth) return <span className="text-secondary">系统状态检查中...</span>;
        const status = systemHealth.overall_api_status || 'Unknown';
        const color = status === 'healthy' ? 'var(--success)' : 'var(--danger)';
        return <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: color, boxShadow: `0 0 8px ${color}` }}></span><span style={{ color, fontWeight: 600, fontSize: '0.875rem' }}>{status.toUpperCase()}</span></div>;
    };

    const renderResults = () => {
        if (!results) return null;
        const isFake = results.is_likely_deepfake;
        const probability = Number(results.deepfake_probability || 0);
        const confidence = (isFake ? probability : (1 - probability)) * 100;
        const risk = probability >= 0.7 ? '高风险' : probability >= 0.4 ? '中风险' : '低风险';
        const verdictColor = isFake ? 'var(--danger)' : 'var(--success)';
        const verdictText = isFake ? '疑似伪造' : '内容真实';
        const advice = probability >= 0.7 ? '建议立即转人工复审并保留取证。' : probability >= 0.4 ? '建议二次检测并结合业务场景人工核验。' : '可正常通过，建议抽样复核。';

        const barData = {
            labels: Object.keys(results.model_results || {}),
            datasets: [{ label: '伪造概率(%)', data: Object.values(results.model_results || {}).map(r => (r.probability ? r.probability * 100 : 0)), backgroundColor: 'rgba(59,130,246,0.6)', borderColor: 'rgba(59,130,246,1)', borderWidth: 1, borderRadius: 4 }],
        };
        const doughnutData = {
            labels: ['真实', '伪造'],
            datasets: [{ data: [100 - probability * 100, probability * 100], backgroundColor: ['rgba(16,185,129,0.6)', 'rgba(239,68,68,0.6)'], borderColor: ['rgba(16,185,129,1)', 'rgba(239,68,68,1)'], borderWidth: 1 }],
        };

        return <div className="animate-fade-in" style={{ marginTop: '2rem' }}>
            <div className="card neon-border" style={{ marginBottom: '1.5rem' }}>
                <h2>检测结论</h2>
                <div style={{ fontSize: '2rem', fontWeight: 800, color: verdictColor }}>{verdictText}</div>
                <p>伪造概率：<strong>{(probability * 100).toFixed(1)}%</strong></p>
                <p>结论置信度：<strong>{confidence.toFixed(1)}%</strong></p>
                <p>风险等级：<strong style={{ color: verdictColor }}>{risk}</strong></p>
                <p>审核建议：{advice}</p>
            </div>

            <div className="card" style={{ marginBottom: '1.5rem' }}>
                <h3>模型检测明细</h3>
                <ul style={{ marginTop: '0.75rem', paddingLeft: '1.2rem' }}>
                    {Object.entries(results.model_results || {}).map(([name, detail]) => <li key={name}>{name}：{((detail.probability || 0) * 100).toFixed(1)}%</li>)}
                </ul>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
                <div className="card"><h3>综合得分图</h3><div style={{ height: 250 }}><Doughnut data={doughnutData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } } }} /></div></div>
                <div className="card"><h3>模型得分图表</h3><div style={{ height: 250 }}><Bar data={barData} options={{ maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, x: { grid: { display: false }, ticks: { color: '#94a3b8' } } }, plugins: { legend: { display: false } } }} /></div></div>
            </div>
        </div>;
    };

    return <div className="App tech-bg">
        <header className="topbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><div className="logo-icon">智鉴</div><h1 style={{ fontSize: '1.35rem' }}>智鉴 DeepSafe-CN</h1></div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>{renderHealthStatus()}<button onClick={onLogout} className="btn btn-secondary">退出登录</button></div>
        </header>
        <main className="container">
            <div style={{ textAlign: 'center', marginBottom: '2rem', paddingTop: '1rem' }}><h2 className="text-gradient" style={{ fontSize: '2.3rem' }}>多模态 AIGC 内容真实性检测平台</h2></div>
            <div className="card glass" style={{ maxWidth: 880, margin: '0 auto', borderStyle: 'dashed', borderWidth: 2 }}>
                <input type="file" id="file-upload" style={{ display: 'none' }} onChange={handleFileChange} accept="image/*,video/*,audio/*" />
                <label htmlFor="file-upload" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
                    <span style={{ fontSize: '1.1rem', fontWeight: 600 }}>{selectedFile ? selectedFile.name : '点击上传或拖拽文件到此处'}</span>
                    <span style={{ color: 'var(--text-secondary)' }}>支持格式：JPG / PNG / MP4 / AVI / WAV</span>
                </label>
            </div>
            {previewUrl && <div className="animate-fade-in" style={{ marginTop: '1.5rem', maxWidth: 880, marginLeft: 'auto', marginRight: 'auto' }}>
                <div className="card glass" style={{ overflow: 'hidden', padding: 0 }}>
                    <div style={{ width: '100%', height: 380, backgroundColor: '#000' }}>{mediaType === 'video' ? <video src={previewUrl} controls style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : mediaType === 'audio' ? <audio src={previewUrl} controls style={{ width: '100%', marginTop: '170px' }} /> : <img src={previewUrl} alt="预览" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />}</div>
                    <div style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}><h3>文件已就绪，开始检测</h3><div style={{ display: 'flex', gap: '0.5rem' }}><button className="btn btn-secondary" onClick={() => loadDemoFile('/demo_video.mp4', 'demo_video.mp4', 'video/mp4', 'video')}>演示视频</button><button className="btn btn-secondary" onClick={() => loadDemoFile('/demo_audio.wav', 'demo_audio.wav', 'audio/wav', 'audio')}>演示音频</button><button className="btn btn-primary" onClick={handleAnalyze} disabled={isAnalyzing}>{isAnalyzing ? '检测中...' : '开始检测'}</button></div></div>
                </div>
            </div>}
            {error && <div className="error-message" style={{ maxWidth: 880, margin: '1rem auto 0' }}>{error}</div>}
            <div style={{ maxWidth: 880, margin: '0 auto' }}>{renderResults()}</div>
        </main>
    </div>;
}

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(() => localStorage.getItem('token') !== null);
    const handleLogin = (token) => { localStorage.setItem('token', token); setIsAuthenticated(true); };
    const handleLogout = () => { localStorage.clear(); setIsAuthenticated(false); };
    return <Router><Routes><Route path="/login" element={<Login onLogin={handleLogin} />} /><Route path="/register" element={<Register onLogin={handleLogin} />} /><Route path="/" element={<ProtectedRoute isAuthenticated={isAuthenticated}><Dashboard onLogout={handleLogout} /></ProtectedRoute>} /></Routes></Router>;
}

export default App;
