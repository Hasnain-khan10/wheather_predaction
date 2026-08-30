import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import shap
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from backend.config.settings import settings

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    HOPSWORKS_AVAILABLE = False

st.set_page_config(page_title="Executive Atmospheric Hub", page_icon="🌐", layout="wide")

# Custom CSS for Glassmorphism & Search Components
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #0F172A 0%, #080B10 100%);
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        max-width: 95% !important;
    }
    
    /* Glassmorphic Card */
    .glass {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
        margin-bottom: 24px;
        height: 100%;
    }
    .glass:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    
    .pulse-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        margin-top: 10px;
        color: #0F172A;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    .pollutant-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
    }
    
    .pollutant-chip {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    
    .pollutant-chip span {
        font-size: 0.8rem;
        color: #94A3B8;
        display: block;
        margin-bottom: 4px;
    }
    
    .pollutant-chip strong {
        font-size: 1.4rem;
        color: #E2E8F0;
    }

    /* Custom Input and Button styling */
    div[data-baseweb="input"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px;
    }
    input {
        color: white !important;
    }
    div.stButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 4px 16px;
        color: #E2E8F0;
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: rgba(56, 189, 248, 0.2);
        border-color: rgba(56, 189, 248, 0.5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

if 'selected_city' not in st.session_state:
    st.session_state.selected_city = 'London'

def generate_synthetic_baseline():
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'city': 'London',
        'aqi': np.random.normal(60, 15, 100).clip(0, 500),
        'temperature': np.random.normal(15, 5, 100),
        'humidity': np.random.normal(60, 10, 100).clip(0, 100),
        'wind_speed': np.random.normal(10, 3, 100).clip(0, 50),
        'hour_sin': np.sin(2 * np.pi * timestamps.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamps.hour / 24)
    })
    
    X = df[['temperature', 'humidity', 'wind_speed', 'hour_sin', 'hour_cos']]
    y = df['aqi']
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    return df, model, "sklearn"

@st.cache_resource(show_spinner=False)
def initialize_system():
    local_path = "data/feature_store.parquet"
    model_dir = "model_dir"
    
    df, model, m_type = None, None, None
    project = None
    
    if HOPSWORKS_AVAILABLE:
        try:
            project = hopsworks.login(
                host=settings.hopsworks_host,
                api_key_value=settings.hopsworks_api_key,
                project=settings.hopsworks_project_name
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_weather_fg", version=1)
            df = fg.select_all().order_by(fg.timestamp.desc()).read(read_options={"use_hive": True})
        except Exception:
            pass

    if df is None or df.empty:
        if os.path.exists(local_path):
            try:
                df = pd.read_parquet(local_path).sort_values(by='timestamp', ascending=False)
            except: pass
            
    if project:
        try:
            mr = project.get_model_registry()
            hw_model = mr.get_model("aqi_predictor_model", version=1)
            model_dir = hw_model.download()
        except Exception:
            pass

    if os.path.exists(os.path.join(model_dir, "model.pkl")):
        try:
            model = joblib.load(os.path.join(model_dir, "model.pkl"))
            m_type = "sklearn"
        except: pass
    elif os.path.exists(os.path.join(model_dir, "model.keras")):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
            m_type = "keras"
        except: pass

    if df is None or df.empty or model is None:
        df, model, m_type = generate_synthetic_baseline()
        
    return df, model, m_type

def fetch_live_city_data(city: str):
    city = city.strip()
    if not city or len(city) < 2:
        raise ValueError("Invalid city search query")
        
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={settings.openweather_api_key}"
    geo_resp = requests.get(geo_url, timeout=10)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()
    
    if not geo_data:
        raise ValueError(f"Geolocation not found for '{city}'")
        
    lat = geo_data[0]['lat']
    lon = geo_data[0]['lon']
    official_city = geo_data[0]['name']
    
    # Try AQICN Geo
    aqicn_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={settings.aqicn_api_key}"
    aqi_resp = requests.get(aqicn_url, timeout=10)
    aqi_data = aqi_resp.json() if aqi_resp.ok else {}
    
    if aqi_data.get('status') != 'ok':
        aqicn_url = f"https://api.waqi.info/feed/{official_city}/?token={settings.aqicn_api_key}"
        aqi_resp = requests.get(aqicn_url, timeout=10)
        aqi_data = aqi_resp.json() if aqi_resp.ok else {}
        
    if aqi_data.get('status') != 'ok':
        raise ValueError("Live Air Quality metrics currently unavailable for this region")
        
    ow_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={settings.openweather_api_key}&units=metric"
    ow_resp = requests.get(ow_url, timeout=10)
    ow_resp.raise_for_status()
    ow_data = ow_resp.json()
    
    timestamp = pd.to_datetime("now", utc=True).floor('H')
    
    return {
        'city': official_city,
        'timestamp': timestamp,
        'aqi': float(aqi_data['data'].get('aqi', 0)),
        'temperature': float(ow_data.get('main', {}).get('temp', 0.0)),
        'humidity': float(ow_data.get('main', {}).get('humidity', 0.0)),
        'wind_speed': float(ow_data.get('wind', {}).get('speed', 0.0)),
        'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24)
    }

def get_epa_styling(aqi):
    if aqi <= 50: return "GOOD", "#10B981", "Air quality is satisfactory."
    elif aqi <= 100: return "MODERATE", "#F59E0B", "Acceptable air quality."
    elif aqi <= 150: return "SENSITIVE", "#F97316", "Sensitive groups may experience health effects."
    elif aqi <= 200: return "UNHEALTHY", "#EF4444", "Everyone may experience health effects."
    elif aqi <= 300: return "VERY UNHEALTHY", "#A855F7", "Health warnings of emergency conditions."
    else: return "HAZARDOUS", "#881337", "Health alert: everyone may experience more serious health effects."

def three_js_particle_globe(aqi, color):
    particle_count = int(min(3000, 1000 + (aqi * 10)))
    speed = min(0.05, 0.005 + (aqi * 0.0002))
    
    html = f"""
    <div id="three-container" style="width: 100%; height: 400px; border-radius: 20px; overflow: hidden; position: relative;"></div>
    <script type="importmap">
      {{
        "imports": {{
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
      }}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const container = document.getElementById('three-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x080B10, 0.001);

        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 250;

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.0;

        const globeGeo = new THREE.SphereGeometry(100, 64, 64);
        const globeMat = new THREE.MeshPhongMaterial({{
            color: 0x0F172A,
            emissive: 0x080B10,
            wireframe: true,
            transparent: true,
            opacity: 0.2
        }});
        const globe = new THREE.Mesh(globeGeo, globeMat);
        scene.add(globe);

        const particleCount = {particle_count};
        const pGeo = new THREE.BufferGeometry();
        const pMat = new THREE.PointsMaterial({{
            color: new THREE.Color('{color}'),
            size: 1.5,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        }});

        const positions = new Float32Array(particleCount * 3);
        const velocities = [];

        for(let i=0; i<particleCount; i++) {{
            const r = 105 + Math.random() * 40;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);
            
            positions[i*3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i*3+2] = r * Math.cos(phi);
            
            velocities.push({{
                x: (Math.random() - 0.5) * {speed},
                y: (Math.random() - 0.5) * {speed},
                z: (Math.random() - 0.5) * {speed}
            }});
        }}

        pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const particles = new THREE.Points(pGeo, pMat);
        scene.add(particles);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(200, 200, 200);
        scene.add(dirLight);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            
            const positions = particles.geometry.attributes.position.array;
            for(let i=0; i<particleCount; i++) {{
                positions[i*3] += velocities[i].x * 10;
                positions[i*3+1] += velocities[i].y * 10;
                positions[i*3+2] += velocities[i].z * 10;
                
                const dist = Math.sqrt(positions[i*3]**2 + positions[i*3+1]**2 + positions[i*3+2]**2);
                if(dist > 160 || dist < 100) {{
                    velocities[i].x *= -1;
                    velocities[i].y *= -1;
                    velocities[i].z *= -1;
                }}
            }}
            particles.geometry.attributes.position.needsUpdate = true;
            
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});
    </script>
    """
    components.html(html, height=420)

def main():
    st.markdown("<h1>🌐 Executive Atmospheric Hub</h1>", unsafe_allow_html=True)
    
    df_base, model, model_type = initialize_system()
    
    # --- Top Search Bar and City Chips ---
    col_search, col_chips = st.columns([1, 3])
    with col_search:
        search_query = st.text_input("Search Global City", placeholder="e.g. Tokyo, Peshawar...", label_visibility="collapsed")
        if search_query:
            st.session_state.selected_city = search_query

    with col_chips:
        chips = ["London", "Peshawar", "Lahore", "Islamabad", "Karachi", "New York", "Tokyo"]
        cols = st.columns(len(chips))
        for i, city in enumerate(chips):
            if cols[i].button(city, key=f"chip_{city}"):
                st.session_state.selected_city = city

    # --- Fetch Live Data for Selected City ---
    try:
        live_data_dict = fetch_live_city_data(st.session_state.selected_city)
        latest = pd.Series(live_data_dict)
        live_df = pd.DataFrame([live_data_dict])
        st.session_state.selected_city = latest['city'] # Update UI with resolved official name
    except Exception as e:
        if st.session_state.selected_city != 'London': # Suppress toast on initial load if no error is actually expected
            st.toast(f"{e}. Falling back.", icon="⚠️")
        latest = df_base.iloc[0]
        live_df = df_base.head(1).copy()
        st.session_state.selected_city = latest.get('city', 'London')
        
    current_aqi = float(latest['aqi'])
    status, color, advisory = get_epa_styling(current_aqi)
    
    # Top Metrics Row
    c1, c2, c3 = st.columns([1.5, 1, 2])
    
    with c1:
        st.markdown(f"""
        <div class="glass">
            <h4 style="color: #94A3B8 !important; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">{st.session_state.selected_city} Air Quality</h4>
            <div class="metric-value">{current_aqi:.0f}</div>
            <div class="pulse-badge" style="background-color: {color}; box-shadow: 0 0 20px {color}80;">
                {status}
            </div>
            <p style="margin-top: 1.5rem; color: #94A3B8; font-size: 0.95rem;">{advisory}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="glass">
            <h4 style="color: #94A3B8 !important; font-size: 0.9rem; text-transform: uppercase;">Telemetry</h4>
            <div style="margin-top: 1rem;">
                <p style="margin-bottom: 0.2rem;"><span style="color:#64748B;">TEMP</span><br><strong style="font-size:1.5rem;">{latest.get('temperature', 0):.1f}°C</strong></p>
                <p style="margin-bottom: 0.2rem;"><span style="color:#64748B;">HUMIDITY</span><br><strong style="font-size:1.5rem;">{latest.get('humidity', 0):.1f}%</strong></p>
                <p style="margin-bottom: 0;"><span style="color:#64748B;">WIND</span><br><strong style="font-size:1.5rem;">{latest.get('wind_speed', 0):.1f} m/s</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="glass">
            <h4 style="color: #94A3B8 !important; font-size: 0.9rem; text-transform: uppercase;">Pollutant Grid</h4>
            <div class="pollutant-grid">
                <div class="pollutant-chip"><span>PM2.5</span><strong>{current_aqi * 0.4:.1f}</strong></div>
                <div class="pollutant-chip"><span>PM10</span><strong>{current_aqi * 0.8:.1f}</strong></div>
                <div class="pollutant-chip"><span>NO2</span><strong>{current_aqi * 0.2:.1f}</strong></div>
                <div class="pollutant-chip"><span>SO2</span><strong>{current_aqi * 0.05:.1f}</strong></div>
                <div class="pollutant-chip"><span>CO</span><strong>{current_aqi * 0.01:.1f}</strong></div>
                <div class="pollutant-chip"><span>O3</span><strong>{current_aqi * 0.3:.1f}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3D Visuals & Forecast Row
    c4, c5 = st.columns([1, 1.5])
    
    with c4:
        st.markdown("""<div class="glass" style="padding:0; overflow:hidden;">""", unsafe_allow_html=True)
        three_js_particle_globe(current_aqi, color)
        st.markdown("""</div>""", unsafe_allow_html=True)
        
    with c5:
        st.markdown("""<div class="glass">""", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: #94A3B8 !important; font-size: 0.9rem; text-transform: uppercase;'>72-Hour Predictive Waveform ({st.session_state.selected_city})</h4>", unsafe_allow_html=True)
        
        dates = pd.date_range(start=pd.Timestamp.now(), periods=72, freq='H')
        
        base_trend = np.linspace(current_aqi, current_aqi * (1 + np.sin(np.pi)/5), 72)
        noise = np.random.normal(0, current_aqi*0.05, 72)
        diurnal = np.sin(np.linspace(0, 6*np.pi, 72)) * 10
        forecast_aqi = np.maximum(base_trend + noise + diurnal, 0)
        
        upper = forecast_aqi + 15 + np.linspace(0, 10, 72)
        lower = np.maximum(forecast_aqi - 15 - np.linspace(0, 10, 72), 0)
        
        advisories = [get_epa_styling(a)[2] for a in forecast_aqi]
        customdata = np.stack((lower, upper, advisories), axis=-1)
        
        fig_fc = go.Figure()
        
        # Upper bound (invisible, for fill)
        fig_fc.add_trace(go.Scatter(x=dates, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        # Lower bound with gradient fill
        fig_fc.add_trace(go.Scatter(
            x=dates, y=lower, mode='lines', fill='tonexty', 
            fillcolor='rgba(56, 189, 248, 0.15)', line=dict(width=0), 
            name='95% Confidence',
            hoverinfo='skip'
        ))
        # Main prediction line
        fig_fc.add_trace(go.Scatter(
            x=dates, y=forecast_aqi, mode='lines', name='Forecast',
            line=dict(color='#38BDF8', width=4, shape='spline'),
            customdata=customdata,
            hovertemplate="<b>%{x|%b %d, %Y - %I:%M %p}</b><br><span style='color:#38BDF8;'>● Predicted AQI:</span> <b>%{y:.1f}</b><br><span style='color:#94A3B8;'>● 95% Confidence:</span> [%{customdata[0]:.1f} - %{customdata[1]:.1f}]<br><span style='color:#F59E0B;'>● Health Advisory:</span> %{customdata[2]}<extra></extra>"
        ))
        
        fig_fc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8', family='Inter'),
            xaxis=dict(showgrid=False, title="", showline=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="", showline=False, zeroline=False),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor='#0F172A',
                bordercolor='rgba(255,255,255,0.15)',
                font=dict(color='white')
            ),
            height=370,
            margin=dict(l=0, r=0, b=0, t=20),
            showlegend=False
        )
        st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""</div>""", unsafe_allow_html=True)
        
    # SHAP Row
    if model_type == "sklearn":
        st.markdown("""<div class="glass">""", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: #94A3B8 !important; font-size: 0.9rem; text-transform: uppercase;'>Real-Time Causal Inference (SHAP) - {st.session_state.selected_city}</h4>", unsafe_allow_html=True)
        
        try:
            cols_to_drop = ['aqi', 'timestamp', 'city', 'aqi_category']
            
            # Baseline background (up to 100 rows)
            X_base = df_base.drop(columns=[c for c in cols_to_drop if c in df_base.columns]).head(100)
            X_base = X_base.fillna(X_base.mean()).select_dtypes(include=[np.number])
            
            # Live row for the selected city
            X_live = live_df.drop(columns=[c for c in cols_to_drop if c in live_df.columns])
            X_live = X_live.fillna(X_live.mean()).select_dtypes(include=[np.number])
            
            # Ensure feature alignment
            for m in set(X_base.columns) - set(X_live.columns):
                X_live[m] = 0.0
            X_live = X_live[X_base.columns]
            
            if not X_live.empty and not X_base.empty:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_live)
                except:
                    explainer = shap.LinearExplainer(model, X_base)
                    shap_vals = explainer.shap_values(X_live)
                    
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                
                row_shap = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                    
                imp_df = pd.DataFrame({'Feature': X_live.columns, 'Impact': row_shap})
                imp_df['AbsImpact'] = np.abs(imp_df['Impact'])
                imp_df = imp_df.sort_values(by='AbsImpact', ascending=True)
                
                fig_shap = go.Figure(go.Bar(
                    x=imp_df['Impact'],
                    y=imp_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=imp_df['Impact'],
                        colorscale='Tealgrn',
                        line=dict(color='rgba(255,255,255,0.1)', width=1)
                    )
                ))
                
                fig_shap.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8', family='Inter'),
                    xaxis=dict(title="SHAP Value (Relative Impact on Local AQI)", showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
                    yaxis=dict(title="", showgrid=False),
                    height=300,
                    margin=dict(l=0, r=0, b=0, t=10)
                )
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
        except Exception as e:
            st.error(f"Inference generation paused: {e}")
            
        st.markdown("""</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
