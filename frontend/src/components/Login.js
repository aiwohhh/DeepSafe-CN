import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API_BASE_URL = window.location.hostname === 'localhost' && window.location.port === '3000'
  ? 'http://localhost:8000'
  : '/api';

const Login = ({ onLogin }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);
            const response = await fetch(`${API_BASE_URL}/token`, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('用户名或密码错误');
            const data = await response.json();
            onLogin(data.access_token);
            navigate('/');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return <div className="auth-container"><div className="card auth-card"><div className="auth-header"><div className="logo-icon">智鉴</div><h2>欢迎回来</h2><p>登录智鉴 DeepSafe-CN 平台</p></div><form onSubmit={handleSubmit}><div className="form-group"><label>用户名</label><input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required className="input-field" placeholder="请输入用户名" /></div><div className="form-group"><label>密码</label><input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required className="input-field" placeholder="请输入密码" /></div>{error && <div className="error-message">{error}</div>}<button type="submit" className="btn btn-primary w-full" disabled={loading}>{loading ? '登录中...' : '登录'}</button></form><div className="auth-footer"><p>还没有账号？ <span onClick={() => navigate('/register')} className="link">去注册</span></p></div></div></div>;
};

export default Login;
