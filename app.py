import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="CrackScan AI Pro", layout="wide", page_icon="🛡️")

# Custom CSS for modern UI and Thumbnail sizing
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #007bff; }
    .stDataFrame { border: 1px solid #e6e9ef; border-radius: 10px; }
    .status-tag { padding: 2px 8px; border-radius: 5px; font-weight: bold; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ CrackScan Control")
    st.subheader("Detection Settings")
    sensitivity = st.slider("Sensitivity", 20, 200, 100)
    min_area = st.number_input("Min Crack Size", value=100)
    st.divider()
    st.info("Grid View Mode Enabled")

# --- 3. MAIN HEADER ---
st.title("Infrastructure Health AI")

# --- 4. UPLOAD & PROCESSING ---
uploaded_files = st.file_uploader("Upload Inspection Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    
    for uploaded_file in uploaded_files:
        # Image Loading
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Edge Detection Logic
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, sensitivity, sensitivity * 2.5)
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Risk Logic
        density = (np.sum(dilated > 0) / dilated.size) * 100
        if density < 0.1:
            status, color = "SAFE", (0, 255, 0)
        elif density < 1.0:
            status, color = "WARNING", (255, 255, 0)
        else:
            status, color = "CRITICAL", (255, 0, 0)
            
        # Draw Labeled Bounding Boxes
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img_rgb.copy()
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                cv2.putText(overlay, f"Crack {count+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                count += 1
        
        results_list.append({
            "name": uploaded_file.name,
            "density": round(density, 4),
            "count": count,
            "status": status,
            "img": overlay 
        })

    df = pd.DataFrame(results_list)

    # --- 5. TABS ---
    tab1, tab2 = st.tabs(["📊 Summary Report", "🖼️ Grid Inspection"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Images", len(df))
        c2.metric("Critical Risks", len(df[df['status'] == 'CRITICAL']))
        c3.metric("Avg Density", f"{df['density'].mean():.2f}%")
        
        st.dataframe(df.drop(columns=['img']), use_container_width=True)

    with tab2:
        st.subheader("Detection Grid")
        # --- GRID LOGIC ---
        # Displaying 3 images per row for better UX
        cols_per_row = 3
        for i in range(0, len(results_list), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(results_list):
                    res = results_list[i + j]
                    with cols[j]:
                        # Display Thumbnail
                        st.image(res["img"], use_container_width=True)
                        
                        # Compact Info under image
                        st.markdown(f"**{res['name']}**")
                        
                        if res['status'] == "CRITICAL":
                            st.error(f"🔴 CRITICAL ({res['density']}%)")
                        elif res['status'] == "WARNING":
                            st.warning(f"🟡 WARNING ({res['density']}%)")
                        else:
                            st.success(f"🟢 SAFE")
                        
                        # Small download button for the specific image
                        res_pil = Image.fromarray(res["img"])
                        buf = io.BytesIO()
                        res_pil.save(buf, format="PNG")
                        st.download_button(label="💾 Save", data=buf.getvalue(), 
                                         file_name=f"result_{res['name']}", key=f"dl_{i+j}",
                                         use_container_width=True)
else:
    st.info("Upload images to view the interactive inspection grid.")