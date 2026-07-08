import streamlit as st
import cv2
import numpy as np
import pywt
#import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
import math
import auth

# Page configuration
st.set_page_config(
    page_title="Digital Watermarking System",
    page_icon="🔐",
    layout="wide"
)

auth.init_db()

# =============================================================================
# Authentication Gate
# =============================================================================
def render_login_page():
    st.title("🔐 Digital Watermarking System")
    st.markdown("### Please log in to continue")

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Log In", type="primary")
            if login_submitted:
                user = auth.verify_user(login_username, login_password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user["username"]
                    st.session_state["role"] = user["role"]
                    auth.log_activity(user["username"], "login")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submitted = st.form_submit_button("Create Account", type="primary")
            if signup_submitted:
                if new_password != confirm_password:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = auth.create_user(new_username, new_password, role="user")
                    if ok:
                        auth.log_activity(new_username, "signup")
                        st.success(f"✅ {msg} You can now log in from the 'Log In' tab.")
                    else:
                        st.error(f"❌ {msg}")

    st.info(
        f"🛠️ Default admin login: **{auth.DEFAULT_ADMIN_USERNAME} / {auth.DEFAULT_ADMIN_PASSWORD}** "
        "— please change this password after your first login (sidebar → Change my password)."
    )


if not st.session_state.get("authenticated"):
    render_login_page()
    st.stop()

# Title and description
st.title("🔐 Digital Watermarking System")
st.markdown("### Advanced Image Watermarking with Multiple Transform Techniques")

# Utility Functions
def calculate_psnr(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return cv2.PSNR(img1, img2)

def calculate_nc(original_wm, extracted_wm):
    """Calculate Normalized Correlation"""
    if len(original_wm.shape) == 3:
        original_wm = cv2.cvtColor(original_wm, cv2.COLOR_BGR2GRAY)
    if len(extracted_wm.shape) == 3:
        extracted_wm = cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2GRAY)
    
    if original_wm.shape != extracted_wm.shape:
        extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))
    
    o = original_wm.astype(np.float32)
    e = extracted_wm.astype(np.float32)
    o -= np.mean(o)
    e -= np.mean(e)
    
    return np.sum(o * e) / np.sqrt(np.sum(o**2) * np.sum(e**2))

def calculate_ber(original_wm, extracted_wm):
    """Calculate Bit Error Rate"""
    if len(original_wm.shape) == 3:
        original_wm = cv2.cvtColor(original_wm, cv2.COLOR_BGR2GRAY)
    if len(extracted_wm.shape) == 3:
        extracted_wm = cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2GRAY)
    
    if original_wm.shape != extracted_wm.shape:
        extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))
    
    original_bin = (original_wm > 127)
    extracted_bin = (extracted_wm > 127)
    errors = np.sum(original_bin != extracted_bin)
    total = original_bin.size
    
    return (errors / total) * 100, errors, total

def interpret_psnr(score):
    """Interpret PSNR score"""
    if score >= 40:
        return "Excellent Quality (Weak/Negligible Distortion)"
    elif 30 <= score < 40:
        return "Good Quality (Moderate/Acceptable Distortion)"
    elif 20 <= score < 30:
        return "Poor/Fair Quality (Strong/Heavy Distortion)"
    else:
        return "Very Poor Quality (Very Heavy/Unacceptable Distortion)"

# =============================================================================
# DCT+DWT Method (Non-blind - needs keys)
# =============================================================================
class WatermarkerDCTDWT:
    def __init__(self, alpha=50.0):
        self.alpha = alpha
        self.block_size = 2

    def text_to_watermark_image(self, text, target_shape):
        h_img, w_img = target_shape
        img = np.zeros((h_img, w_img), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, thickness)
        target_w = w_img * 0.9
        target_h = h_img * 0.9
        scale_x = target_w / text_w if text_w > 0 else 1
        scale_y = target_h / text_h if text_h > 0 else 1
        final_scale = min(scale_x, scale_y)
        (new_w, new_h), _ = cv2.getTextSize(text, font, final_scale, thickness)
        x = int((w_img - new_w) / 2)
        y = int((h_img + new_h) / 2)
        cv2.putText(img, text, (x, y), font, final_scale, (255), thickness)
        return img

    def embed(self, cover, text_string):
        if len(cover.shape) == 3:
            cover_ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(cover_ycrcb)
        else:
            y = cover
            cr = cb = None
            
        y_float = y.astype(np.float32)
        coeffs = pywt.dwt2(y_float, 'haar')
        LL, (LH, HL, HH) = coeffs
        h_ll, w_ll = LL.shape

        watermark = self.text_to_watermark_image(text_string, (h_ll, w_ll))
        wm_norm = watermark.astype(np.float32) / 255.0

        LL_new = np.copy(LL)
        dct_map = {}
        
        for i in range(0, h_ll, self.block_size):
            for j in range(0, w_ll, self.block_size):
                block = LL[i:i+self.block_size, j:j+self.block_size]
                if block.shape != (self.block_size, self.block_size):
                    continue
                block_dct = cv2.dct(block)
                current_dc = block_dct[0, 0]
                dct_map[(i,j)] = current_dc
                block_dct[0, 0] = current_dc + (self.alpha * wm_norm[i, j])
                LL_new[i:i+self.block_size, j:j+self.block_size] = cv2.idct(block_dct)

        coeffs_new = (LL_new, (LH, HL, HH))
        y_watermarked = pywt.idwt2(coeffs_new, 'haar')
        y_final = np.clip(y_watermarked, 0, 255).astype(np.uint8)
        
        if cr is not None and cb is not None:
            img_final = cv2.merge([y_final, cr, cb])
            output_img = cv2.cvtColor(img_final, cv2.COLOR_YCrCb2BGR)
        else:
            output_img = y_final
        
        return output_img, watermark, dct_map, (h_ll, w_ll)

    def extract(self, watermarked_img, dct_map, shape):
        if len(watermarked_img.shape) == 3:
            y = cv2.split(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb))[0].astype(np.float32)
        else:
            y = watermarked_img.astype(np.float32)
            
        coeffs = pywt.dwt2(y, 'haar')
        LL, _ = coeffs
        h_ll, w_ll = shape
        
        extracted_raw = np.zeros((h_ll, w_ll), np.float32)
        
        for i in range(0, h_ll, self.block_size):
            for j in range(0, w_ll, self.block_size):
                block = LL[i:i+self.block_size, j:j+self.block_size]
                if block.shape != (self.block_size, self.block_size):
                    continue
                block_dct = cv2.dct(block)
                if (i,j) in dct_map:
                    dct_old = dct_map[(i,j)]
                    dct_new = block_dct[0, 0]
                    val = (dct_new - dct_old) / self.alpha
                    extracted_raw[i, j] = val

        raw_norm = cv2.normalize(extracted_raw, None, 0, 255, cv2.NORM_MINMAX)
        raw_uint8 = raw_norm.astype(np.uint8)
        large_view = cv2.resize(raw_uint8, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((2,2), np.uint8)
        connected_view = cv2.dilate(large_view, kernel, iterations=1)
        return connected_view

# =============================================================================
# Pure DWT Method (Non-blind - needs original image)
# =============================================================================
class WatermarkerDWT:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def text_to_image(self, text, width, height, font_scale=1, thickness=2):
        img = np.ones((height, width), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            if size[0] < (width - 10) and size[1] < (height - 10):
                break
            font_scale -= 0.1
            if font_scale < 0.1:
                font_scale = 0.1
                break
        x = max((width - size[0]) // 2, 0)
        y = max((height + size[1]) // 2, size[1])
        cv2.putText(img, text, (x, y), font, font_scale, (0,), thickness)
        return img

    def embed(self, host, text):
        if len(host.shape) == 3:
            host_ycrcb = cv2.cvtColor(host, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(host_ycrcb)
        else:
            y, cr, cb = host, None, None

        y_float = y.astype(np.float32)
        LL, (LH, HL, HH) = pywt.dwt2(y_float, 'haar')
        h, w = LH.shape
        initial_scale = max(1, w / 200)
        wm_ref = self.text_to_image(text, w, h, font_scale=initial_scale)
        wm_sparse = 255 - wm_ref
        wm_float = wm_sparse.astype(np.float32)
        LH_new = LH + self.alpha * wm_float
        y_watermarked = pywt.idwt2((LL, (LH_new, HL, HH)), 'haar')
        y_final = np.clip(y_watermarked, 0, 255).astype(np.uint8)

        if cr is not None:
            result = cv2.cvtColor(cv2.merge([y_final, cr, cb]), cv2.COLOR_YCrCb2BGR)
        else:
            result = y_final
        return result, wm_ref

    def extract(self, original, watermarked):
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        if len(watermarked.shape) == 3:
            watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        h, w = original.shape
        watermarked = cv2.resize(watermarked, (w, h))
        original = original.astype(np.float32)
        watermarked = watermarked.astype(np.float32)
        _, (LH1, _, _) = pywt.dwt2(original, 'haar')
        _, (LH2, _, _) = pywt.dwt2(watermarked, 'haar')
        diff = (LH2 - LH1) / self.alpha
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        recovered = 255 - binary
        return recovered

# =============================================================================
# DWT-SVD Method (Non-blind - needs original image AND watermark)
# =============================================================================
class WatermarkerDWTSVD:
    def __init__(self, alpha=0.009):
        self.alpha = alpha

    def text_to_image(self, text, width=200, height=50):
        text1 = "Hospital: " + text
        text2 = "DateTime: " + datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        img = np.ones((height, width), dtype=np.uint8) * 255
        text_x1 = int(width//15)
        text_y1 = int(height//10)
        text_x2 = int(width//15)
        text_y2 = int((height//10)*2 + 15)
        cv2.putText(img, text1, (text_x1, text_y1), font, font_scale, (0,), thickness)
        cv2.putText(img, text2, (text_x2, text_y2), font, font_scale, (0,), thickness)
        return img

    def embed(self, cover_img, wm_img):
        if len(cover_img.shape) == 3:
            cover_ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(cover_ycrcb)
        else:
            y, cr, cb = cover_img, None, None

        h, w = y.shape
        if h % 2 != 0:
            h -= 1
        if w % 2 != 0:
            w -= 1
        y = cv2.resize(y, (w, h))
        wm_img = cv2.resize(wm_img, (w, h))
        if cr is not None:
            cr = cv2.resize(cr, (w, h))
            cb = cv2.resize(cb, (w, h))

        y_f = y.astype(np.float32) / 255.0
        wm_f = wm_img.astype(np.float32) / 255.0
        LLori, (LH, HL, HH) = pywt.dwt2(y_f, 'haar')
        LLwm, _ = pywt.dwt2(wm_f, 'haar')
        U, S, Vt = np.linalg.svd(LLori, full_matrices=False)
        _, Swm, _ = np.linalg.svd(LLwm, full_matrices=False)
        S_new = S + self.alpha * Swm
        LL_new = U @ np.diag(S_new) @ Vt
        y_watermarked_f = pywt.idwt2((LL_new, (LH, HL, HH)), 'haar')
        y_watermarked = np.clip(y_watermarked_f * 255.0, 0, 255).astype(np.uint8)

        if cr is not None:
            watermarked = cv2.cvtColor(cv2.merge([y_watermarked, cr, cb]), cv2.COLOR_YCrCb2BGR)
        else:
            watermarked = y_watermarked
        return watermarked, wm_img

    def extract(self, watermarked_img, original_cover, watermark):
        if len(watermarked_img.shape) == 3:
            watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        if len(original_cover.shape) == 3:
            original_cover = cv2.cvtColor(original_cover, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        h, w = watermarked_img.shape[:2]
        if original_cover.shape[:2] != (h, w):
            original_cover = cv2.resize(original_cover, (w, h))
        if watermark.shape[:2] != (h, w):
            watermark = cv2.resize(watermark, (w, h))

        wmed_f = watermarked_img.astype(np.float32) / 255.0
        cover_f = original_cover.astype(np.float32) / 255.0
        wm_f = watermark.astype(np.float32) / 255.0
        LLwmed, (LHwmed, HLwmed, HHwmed) = pywt.dwt2(wmed_f, 'haar')
        LLori, _ = pywt.dwt2(cover_f, 'haar')
        LLwm, _ = pywt.dwt2(wm_f, 'haar')
        _, Swmed, _ = np.linalg.svd(LLwmed, full_matrices=False)
        _, Sori, _ = np.linalg.svd(LLori, full_matrices=False)
        Uwm, Swm, Vwm = np.linalg.svd(LLwm, full_matrices=False)
        Swm_extracted = (Swmed - Sori) / self.alpha
        LLext = Uwm @ np.diag(Swm_extracted) @ Vwm
        wm_ext_f = pywt.idwt2((LLext, (LHwmed, HLwmed, HHwmed)), 'haar')
        return np.clip(wm_ext_f * 255.0, 0, 255).astype(np.uint8)

# =============================================================================
# DCT + Linear Modulation Method (Non-blind - needs original image)
# =============================================================================
class WatermarkerDCTLinear:
    def __init__(self, alpha=0.5, block_size=8):
        self.alpha = alpha
        self.block_size = block_size

    def get_zigzag_order(self):
        zigzag = []
        for diag in range(2 * self.block_size - 1):
            if diag < self.block_size:
                if diag % 2 == 0:
                    for i in range(diag + 1):
                        zigzag.append((diag - i, i))
                else:
                    for i in range(diag + 1):
                        zigzag.append((i, diag - i))
            else:
                if diag % 2 == 0:
                    for i in range(self.block_size - 1, diag - self.block_size, -1):
                        zigzag.append((i, diag - i))
                else:
                    for i in range(self.block_size - 1, diag - self.block_size, -1):
                        zigzag.append((diag - i, i))
        return zigzag

    def get_middle_frequency_position(self):
        zigzag = self.get_zigzag_order()
        middle_positions = zigzag[11:40]
        return middle_positions[9]

    def resize_watermark(self, watermark_img, host_img):
        host_height, host_width = host_img.shape[:2]
        num_blocks_h = host_height // self.block_size
        num_blocks_w = host_width // self.block_size
        total_bits = num_blocks_h * num_blocks_w
        max_size = int(np.sqrt(total_bits))
        
        wm_h, wm_w = watermark_img.shape[:2]
        if wm_h * wm_w > total_bits:
            watermark_resized = cv2.resize(watermark_img, (max_size, max_size), 
                                          interpolation=cv2.INTER_LANCZOS4)
        else:
            watermark_resized = watermark_img
        return watermark_resized

    def image_to_binary(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (image > 127).astype(np.float32).flatten()

    def embed(self, original_img, watermark_img):
        img_ycrcb = cv2.cvtColor(original_img, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycrcb)
        Y = Y.astype(np.float32)
        
        watermark_resized = self.resize_watermark(watermark_img, Y)
        watermark_bits = self.image_to_binary(watermark_resized)
        watermark_h, watermark_w = watermark_resized.shape[:2]
        
        height, width = Y.shape
        watermarked_Y = Y.copy()
        embed_position = self.get_middle_frequency_position()
        WATERMARK_SCALE = 30
        
        bit_index = 0
        total_bits = len(watermark_bits)
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if bit_index >= total_bits:
                    break
                block = Y[i:i+self.block_size, j:j+self.block_size].copy()
                dct_block = cv2.dct(block.astype(np.float32))
                row, col = embed_position
                watermark_bit = float(watermark_bits[bit_index])
                original_coeff = dct_block[row, col]
                watermark_signal = (watermark_bit * 2.0 - 1.0) * WATERMARK_SCALE
                modified_coeff = (1 - self.alpha) * original_coeff + self.alpha * watermark_signal
                dct_block[row, col] = modified_coeff
                bit_index += 1
                reconstructed_block = cv2.idct(dct_block)
                reconstructed_block = np.clip(reconstructed_block, 0, 255)
                watermarked_Y[i:i+self.block_size, j:j+self.block_size] = reconstructed_block
        
        watermarked_Y = watermarked_Y.astype(np.uint8)
        watermarked_ycrcb = cv2.merge([watermarked_Y, Cr, Cb])
        watermarked_img = cv2.cvtColor(watermarked_ycrcb, cv2.COLOR_YCrCb2RGB)
        
        return watermarked_img, watermark_resized, (watermark_h, watermark_w), bit_index

    def binary_to_image(self, binary_array, height, width):
        binary_2d = binary_array.reshape(height, width)
        return (binary_2d * 255).astype(np.uint8)

    def extract(self, original_img, watermarked_img, watermark_dims, num_bits):
        orig_ycrcb = cv2.cvtColor(original_img, cv2.COLOR_RGB2YCrCb)
        wm_ycrcb = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2YCrCb)
        
        Y_orig = orig_ycrcb[:, :, 0].astype(np.float32)
        Y_wm = wm_ycrcb[:, :, 0].astype(np.float32)
        
        height, width = Y_orig.shape
        embed_position = self.get_middle_frequency_position()
        
        extracted_bits = []
        bit_index = 0
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if bit_index >= num_bits:
                    break
                orig_block = Y_orig[i:i+self.block_size, j:j+self.block_size]
                wm_block = Y_wm[i:i+self.block_size, j:j+self.block_size]
                dct_orig = cv2.dct(orig_block.astype(np.float32))
                dct_wm = cv2.dct(wm_block.astype(np.float32))
                row, col = embed_position
                orig_coeff = dct_orig[row, col]
                wm_coeff = dct_wm[row, col]
                extracted_value = (wm_coeff - (1 - self.alpha) * orig_coeff) / self.alpha
                extracted_bit = 1 if extracted_value > 0 else 0
                extracted_bits.append(extracted_bit)
                bit_index += 1
        
        extracted_bits_array = np.array(extracted_bits[:num_bits])
        extracted_watermark = self.binary_to_image(extracted_bits_array, 
                                                   watermark_dims[0], watermark_dims[1])
        return extracted_watermark

# =============================================================================
# Streamlit Interface
# =============================================================================

# Sidebar
with st.sidebar:
    st.markdown(f"👤 Logged in as **{st.session_state['username']}** ({st.session_state['role']})")

    if st.button("🚪 Log Out"):
        auth.log_activity(st.session_state["username"], "logout")
        for key in ("authenticated", "username", "role"):
            st.session_state.pop(key, None)
        st.rerun()

    with st.expander("🔑 Change my password"):
        with st.form("change_password_form"):
            current_pw = st.text_input("Current password", type="password", key="cur_pw")
            new_pw = st.text_input("New password", type="password", key="new_pw")
            confirm_pw = st.text_input("Confirm new password", type="password", key="confirm_pw")
            if st.form_submit_button("Update Password"):
                if not auth.verify_user(st.session_state["username"], current_pw):
                    st.error("❌ Current password is incorrect.")
                elif new_pw != confirm_pw:
                    st.error("❌ New passwords do not match.")
                else:
                    ok, msg = auth.reset_password(st.session_state["username"], new_pw)
                    if ok:
                        auth.log_activity(st.session_state["username"], "self_password_change")
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

    st.markdown("---")
    st.header("⚙️ Settings")
    
    method = st.selectbox(
        "Watermarking Method",
        ["DCT+DWT", "Pure DWT", "DWT+SVD", "DCT+Linear Modulation"]
    )
    
    st.markdown("---")
    
    if method == "DCT+DWT":
        alpha_param = st.slider("Alpha (Embedding Strength)", 10.0, 100.0, 50.0, 5.0)
        st.info("🔑 Non-blind: Requires embedding keys for extraction")
    elif method == "Pure DWT":
        alpha_param = st.slider("Alpha (Embedding Strength)", 0.01, 0.5, 0.1, 0.01)
        st.info("📷 Non-blind: Requires original image for extraction")
    elif method == "DWT+SVD":
        alpha_param = st.slider("Alpha (Embedding Strength)", 0.001, 0.02, 0.009, 0.001)
        st.info("🏥 Non-blind: Requires original image AND watermark for extraction")
    else:  # DCT+Linear Modulation
        alpha_param = st.slider("Alpha (Embedding Strength)", 0.1, 1.0, 0.5, 0.1)
        block_size = st.selectbox("Block Size", [8, 16], index=0)
        st.info("📷 Non-blind: Requires original image for extraction")

# Main tabs
_tab_labels = ["📝 Embed Watermark", "🔍 Extract Watermark", "⚔️ Attack Testing"]
_is_admin = st.session_state.get("role") == "admin"
if _is_admin:
    _tab_labels.append("🛡️ Admin Panel")

_tabs = st.tabs(_tab_labels)
tab1, tab2, tab3 = _tabs[0], _tabs[1], _tabs[2]
tab4 = _tabs[3] if _is_admin else None

# =============================================================================
# Tab 1: Embed Watermark
# =============================================================================
with tab1:
    st.header("Embed Watermark")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Cover Image")
        cover_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'], key='cover_embed')
        
    with col2:
        st.subheader("Watermark Input")
        
        if method == "DCT+Linear Modulation":
            wm_input_type = st.radio("Watermark Type", ["Text", "Image"])
            
            if wm_input_type == "Text":
                watermark_text = st.text_input("Enter watermark text", "Confidential")
            else:
                watermark_file = st.file_uploader("Upload watermark image", 
                                                 type=['png', 'jpg', 'jpeg'], key='watermark_embed')
        else:
            watermark_text = st.text_input("Enter watermark text", "Confidential")
    
    if st.button("🔒 Embed Watermark", type="primary"):
        if cover_file is not None:
            # Load cover image
            cover_bytes = np.asarray(bytearray(cover_file.read()), dtype=np.uint8)
            cover_img = cv2.imdecode(cover_bytes, cv2.IMREAD_COLOR)
            
            embed_extra = {}
            
            # Initialize watermarker based on method
            if method == "DCT+DWT":
                wm = WatermarkerDCTDWT(alpha=alpha_param)
                watermarked, wm_img, dct_map, shape = wm.embed(cover_img, watermark_text)
                
                # Save keys to session for later use (extraction tab convenience)
                st.session_state['embed_dct_map'] = dct_map
                st.session_state['embed_shape'] = shape
                
                keys_data = {'dct_map': dct_map, 'shape': shape}
                import pickle
                embed_extra['keys_bytes'] = pickle.dumps(keys_data)
                
            elif method == "Pure DWT":
                wm = WatermarkerDWT(alpha=alpha_param)
                watermarked, wm_img = wm.embed(cover_img, watermark_text)
                
            elif method == "DWT+SVD":
                wm = WatermarkerDWTSVD(alpha=alpha_param)
                h, w = cover_img.shape[:2]
                if h % 2 != 0:
                    h -= 1
                if w % 2 != 0:
                    w -= 1
                cover_img = cv2.resize(cover_img, (w, h))
                wm_img = wm.text_to_image(watermark_text, w, h)
                watermarked, wm_img = wm.embed(cover_img, wm_img)
                
                _, wm_buffer = cv2.imencode('.png', wm_img)
                embed_extra['wm_buffer'] = wm_buffer.tobytes()
                
            else:  # DCT+Linear Modulation
                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                if wm_input_type == "Text":
                    temp_wm = np.ones((100, 400), dtype=np.uint8) * 255
                    cv2.putText(temp_wm, watermark_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,), 2)
                    wm_img = temp_wm
                else:
                    wm_bytes = np.asarray(bytearray(watermark_file.read()), dtype=np.uint8)
                    wm_img = cv2.imdecode(wm_bytes, cv2.IMREAD_GRAYSCALE)
                
                cover_rgb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
                watermarked, wm_resized, wm_dims, num_bits = wm.embed(cover_rgb, wm_img)
                watermarked = cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR)
            
            # Calculate PSNR
            psnr = calculate_psnr(cover_img, watermarked)
            
            # Encode the watermarked image once so the download button can reuse it
            _, wm_out_buffer = cv2.imencode('.png', watermarked)
            
            # Persist everything needed to redraw this tab across reruns
            # (e.g. reruns triggered by clicking a download button)
            st.session_state['embed_result'] = {
                'method': method,
                'cover_img': cover_img,
                'wm_img': wm_img,
                'watermarked': watermarked,
                'psnr': psnr,
                'watermarked_bytes': wm_out_buffer.tobytes(),
                **embed_extra
            }
            auth.log_activity(
                st.session_state["username"],
                "embed_watermark",
                f"method={method}, psnr={psnr:.2f}dB"
            )
        else:
            st.warning("⚠️ Please upload a cover image first!")
    
    # Render results from session_state so they survive reruns caused by
    # clicking any of the download buttons below
    if 'embed_result' in st.session_state and st.session_state['embed_result']['method'] == method:
        result = st.session_state['embed_result']
        cover_img = result['cover_img']
        wm_img = result['wm_img']
        watermarked = result['watermarked']
        psnr = result['psnr']
        
        st.success(f"✅ Watermark embedded successfully! PSNR: {psnr:.2f} dB")
        st.info(interpret_psnr(psnr))
        
        if 'keys_bytes' in result:
            st.download_button(
                label="📥 Download Embedding Keys (Required for Extraction!)",
                data=result['keys_bytes'],
                file_name="embedding_keys.pkl",
                mime="application/octet-stream",
                key="download_keys_btn"
            )
        
        if 'wm_buffer' in result:
            st.download_button(
                label="📥 Download Watermark Image (Required for Extraction!)",
                data=result['wm_buffer'],
                file_name="watermark_reference.png",
                mime="image/png",
                key="download_wmref_btn"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original")
            if len(cover_img.shape) == 2:
                st.image(cover_img, clamp=True)
            else:
                st.image(cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB))
        
        with col2:
            st.subheader("Watermark")
            if len(wm_img.shape) == 2:
                st.image(wm_img, clamp=True)
            else:
                st.image(cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB))
        
        with col3:
            st.subheader("Watermarked")
            if len(watermarked.shape) == 2:
                st.image(watermarked, clamp=True)
            else:
                st.image(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
        
        # Download watermarked image
        st.download_button(
            label="📥 Download Watermarked Image",
            data=result['watermarked_bytes'],
            file_name="watermarked_image.png",
            mime="image/png",
            key="download_watermarked_btn"
        )

# =============================================================================
# Tab 2: Extract Watermark
# =============================================================================
with tab2:
    st.header("Extract Watermark")
    
    st.info(f"""
    **Current Method: {method}**
    
    Required for extraction:
    - DCT+DWT: Watermarked image + Embedding keys file
    - Pure DWT: Watermarked image + Original cover image
    - DWT+SVD: Watermarked image + Original cover image + Original watermark image
    - DCT+Linear Modulation: Watermarked image + Original cover image
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Watermarked Image")
        watermarked_file = st.file_uploader("Choose watermarked image", 
                                           type=['png', 'jpg', 'jpeg'], key='wm_extract')
    
    with col2:
        if method == "DCT+DWT":
            st.subheader("Upload Embedding Keys")
            keys_file = st.file_uploader("Choose keys file (.pkl)", 
                                        type=['pkl'], key='keys_extract')
        else:
            st.subheader("Upload Original Cover Image")
            original_file = st.file_uploader("Choose original image", 
                                            type=['png', 'jpg', 'jpeg'], key='original_extract')
            
            if method == "DWT+SVD":
                st.subheader("Upload Original Watermark")
                ref_wm_file = st.file_uploader("Choose watermark reference image", 
                                              type=['png', 'jpg', 'jpeg'], key='ref_wm_extract')
    
    if st.button("🔓 Extract Watermark", type="primary"):
        if watermarked_file is not None:
            # Load watermarked image
            wm_bytes = np.asarray(bytearray(watermarked_file.read()), dtype=np.uint8)
            watermarked_img = cv2.imdecode(wm_bytes, cv2.IMREAD_COLOR)
            if watermarked_img is None:
                watermarked_img = cv2.imdecode(wm_bytes, cv2.IMREAD_GRAYSCALE)
            
            extracted = None
            
            try:
                if method == "DCT+DWT":
                    if keys_file is None:
                        st.error("❌ Please upload the embedding keys file!")
                    else:
                        import pickle
                        keys_data = pickle.load(keys_file)
                        dct_map = keys_data['dct_map']
                        shape = keys_data['shape']
                        
                        wm = WatermarkerDCTDWT(alpha=alpha_param)
                        extracted = wm.extract(watermarked_img, dct_map, shape)
                        
                elif method == "Pure DWT":
                    if original_file is None:
                        st.error("❌ Please upload the original cover image!")
                    else:
                        orig_bytes = np.asarray(bytearray(original_file.read()), dtype=np.uint8)
                        original_img = cv2.imdecode(orig_bytes, cv2.IMREAD_COLOR)
                        
                        wm = WatermarkerDWT(alpha=alpha_param)
                        extracted = wm.extract(original_img, watermarked_img)
                        
                elif method == "DWT+SVD":
                    if original_file is None or ref_wm_file is None:
                        st.error("❌ Please upload both original cover image and watermark reference!")
                    else:
                        orig_bytes = np.asarray(bytearray(original_file.read()), dtype=np.uint8)
                        original_img = cv2.imdecode(orig_bytes, cv2.IMREAD_COLOR)
                        
                        ref_wm_bytes = np.asarray(bytearray(ref_wm_file.read()), dtype=np.uint8)
                        ref_watermark = cv2.imdecode(ref_wm_bytes, cv2.IMREAD_GRAYSCALE)
                        
                        wm = WatermarkerDWTSVD(alpha=alpha_param)
                        extracted = wm.extract(watermarked_img, original_img, ref_watermark)
                        
                        # For metric calculation
                        original_wm_for_metrics = ref_watermark
                        
                else:  # DCT+Linear Modulation
                    if original_file is None:
                        st.error("❌ Please upload the original cover image!")
                    else:
                        orig_bytes = np.asarray(bytearray(original_file.read()), dtype=np.uint8)
                        original_img = cv2.imdecode(orig_bytes, cv2.IMREAD_COLOR)
                        
                        # Need to get watermark dimensions - ask user or use default
                        st.warning("⚠️ For DCT+Linear Modulation, watermark dimensions are needed. Using auto-calculated size.")
                        
                        cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        wm_rgb = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB) if len(watermarked_img.shape) == 3 else cv2.cvtColor(cv2.cvtColor(watermarked_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
                        
                        # Calculate dimensions
                        img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                        Y = img_ycrcb[:, :, 0]
                        host_height, host_width = Y.shape[:2]
                        num_blocks_h = host_height // block_size
                        num_blocks_w = host_width // block_size
                        total_bits = num_blocks_h * num_blocks_w
                        max_size = int(np.sqrt(total_bits))
                        wm_dims = (max_size, max_size)
                        num_bits = max_size * max_size
                        
                        wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                        extracted = wm.extract(cover_rgb, wm_rgb, wm_dims, num_bits)
                
                if extracted is not None:
                    st.success("✅ Watermark extracted successfully!")
                    auth.log_activity(
                        st.session_state["username"],
                        "extract_watermark",
                        f"method={method}"
                    )
                    
                    # Display extracted watermark
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Watermarked Image")
                        if len(watermarked_img.shape) == 2:
                            st.image(watermarked_img, clamp=True)
                        else:
                            st.image(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB))
                    
                    with col2:
                        st.subheader("Extracted Watermark")
                        st.image(extracted, clamp=True)
                    
                    # Calculate metrics if reference watermark is available
                    if method == "DWT+SVD" and ref_wm_file is not None:
                        nc = calculate_nc(original_wm_for_metrics, extracted)
                        ber, errors, total = calculate_ber(original_wm_for_metrics, extracted)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Normalized Correlation (NC)", f"{nc:.4f}")
                        with col2:
                            st.metric("Bit Error Rate (BER)", f"{ber:.2f}%")
                    
                    # Download extracted watermark
                    _, ext_buffer = cv2.imencode('.png', extracted)
                    st.download_button(
                        label="📥 Download Extracted Watermark",
                        data=ext_buffer.tobytes(),
                        file_name="extracted_watermark.png",
                        mime="image/png"
                    )
                    
            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
                st.error("Please make sure you uploaded the correct files and used the same parameters as during embedding.")
        else:
            st.warning("⚠️ Please upload a watermarked image!")

# =============================================================================
# Tab 3: Attack Testing
# =============================================================================
with tab3:
    st.header("Attack Testing & Robustness Evaluation")
    
    st.info(f"""
    **Current Method: {method}**
    
    This tab will apply various attacks to your watermarked image and then extract the watermark from each attacked version.
    
    Required files:
    - Watermarked image
    - Same requirements as extraction tab (original image, keys, etc.)
    """)
    
    # Attack selection
    st.subheader("Select Attacks to Test")
    col1, col2 = st.columns(2)
    
    with col1:
        test_rotation = st.checkbox("Rotation", value=True)
        test_jpeg = st.checkbox("JPEG Compression", value=True)
        test_salt_pepper = st.checkbox("Salt & Pepper Noise", value=True)
        test_gaussian = st.checkbox("Gaussian Noise", value=True)
        test_scaling = st.checkbox("Scaling/Resizing", value=True)
    
    with col2:
        test_translation = st.checkbox("Translation", value=False)
        test_cropping = st.checkbox("Cropping", value=False)
        test_flipping = st.checkbox("Flipping", value=False)
        test_filtering = st.checkbox("Filtering (Blur)", value=True)
        test_format = st.checkbox("Format Conversion", value=True)
    
    st.markdown("---")
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Watermarked Image")
        attack_wm_file = st.file_uploader("Choose watermarked image", 
                                         type=['png', 'jpg', 'jpeg'], key='attack_wm')
    
    with col2:
        if method == "DCT+DWT":
            st.subheader("Upload Embedding Keys")
            attack_keys_file = st.file_uploader("Choose keys file (.pkl)", 
                                               type=['pkl'], key='attack_keys')
        else:
            st.subheader("Upload Original Cover Image")
            attack_original_file = st.file_uploader("Choose original image", 
                                                   type=['png', 'jpg', 'jpeg'], key='attack_original')
            
            if method == "DWT+SVD":
                st.subheader("Upload Original Watermark")
                attack_ref_wm_file = st.file_uploader("Choose watermark reference", 
                                                     type=['png', 'jpg', 'jpeg'], key='attack_ref_wm')
    
    # Attack functions
    def apply_rotation_attack(image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    def apply_jpeg_compression(image, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def apply_salt_pepper_noise(image, density):
        noisy = image.copy()
        total_pixels = image.shape[0] * image.shape[1]
        num_salt = int(total_pixels * density / 2)
        num_pepper = int(total_pixels * density / 2)
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        noisy[salt_coords[0], salt_coords[1], :] = 255
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        noisy[pepper_coords[0], pepper_coords[1], :] = 0
        return noisy
    
    def apply_gaussian_noise(image, variance):
        mean = 0
        sigma = variance ** 0.5
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy = image.astype(np.float64) + gaussian * 255
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_scaling_attack(image, scale_factor):
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(scaled, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def apply_translation_attack(image, shift_x, shift_y):
        height, width = image.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    def apply_cropping_attack(image, pct):
        height, width = image.shape[:2]
        crop = int(height * pct / 100)
        cropped = image[crop:height-crop, crop:width-crop]
        padded = np.ones((height, width, 3), dtype=np.uint8) * 255
        crop_height, crop_width = cropped.shape[:2]
        y_offset = (height - crop_height) // 2
        x_offset = (width - crop_width) // 2
        padded[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width] = cropped
        return padded
    
    def apply_flipping_attack(image, flip_type):
        if flip_type == 'horizontal':
            return cv2.flip(image, 1)
        elif flip_type == 'vertical':
            return cv2.flip(image, 0)
        else:
            return cv2.flip(image, -1)
    
    def apply_filtering_attack(image, kernel_size):
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_format_conversion(image, format_type):
        if format_type == 'jpg':
            _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif format_type == 'png':
            _, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif format_type == 'webp':
            _, encoded = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 90])
        else:
            _, encoded = cv2.imencode('.png', image)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    if st.button("🚀 Run Attack Tests", type="primary"):
        if attack_wm_file is None:
            st.error("❌ Please upload a watermarked image!")
        else:
            auth.log_activity(
                st.session_state["username"],
                "run_attack_tests",
                f"method={method}"
            )
            # Load watermarked image
            wm_bytes = np.asarray(bytearray(attack_wm_file.read()), dtype=np.uint8)
            watermarked_img = cv2.imdecode(wm_bytes, cv2.IMREAD_COLOR)
            if watermarked_img is None:
                watermarked_img = cv2.imdecode(wm_bytes, cv2.IMREAD_GRAYSCALE)
            
            # Load required files for extraction
            extraction_ready = False
            
            if method == "DCT+DWT":
                if attack_keys_file is None:
                    st.error("❌ Please upload the embedding keys file!")
                else:
                    import pickle
                    keys_data = pickle.load(attack_keys_file)
                    dct_map = keys_data['dct_map']
                    shape = keys_data['shape']
                    extraction_ready = True
                    
            elif method == "Pure DWT":
                if attack_original_file is None:
                    st.error("❌ Please upload the original cover image!")
                else:
                    orig_bytes = np.asarray(bytearray(attack_original_file.read()), dtype=np.uint8)
                    original_img = cv2.imdecode(orig_bytes, cv2.IMREAD_COLOR)
                    extraction_ready = True
                    
            elif method == "DWT+SVD":
                if attack_original_file is None or attack_ref_wm_file is None:
                    st.error("❌ Please upload both original cover and watermark reference!")
                else:
                    orig_bytes = np.asarray(bytearray(attack_original_file.read()), dtype=np.uint8)
                    original_img = cv2.imdecode(orig_bytes, cv2.IMREAD_COLOR)
                    ref_wm_bytes = np.asarray(bytearray(attack_ref_wm_file.read()), dtype=np.uint8)
                    ref_watermark = cv2.imdecode(ref_wm_bytes, cv2.IMREAD_GRAYSCALE)
                    extraction_ready = True
                    
            else:  # DCT+Linear Modulation
                if attack_original_file is None:
                    st.error("❌ Please upload the original cover image!")
                else:
                    orig_bytes = np.asarray(bytearray(attack_original_file.read()), dtype=np.uint8)
                    original_img = cv2.imdecode(orig_bytes, cv2.IMREAD_COLOR)
                    extraction_ready = True
            
            if extraction_ready:
                st.success("✅ Files loaded. Running attacks...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Store results
                attack_results = []
                total_attacks = 0
                current_attack = 0
                
                # Count total attacks
                if test_rotation: total_attacks += 3
                if test_jpeg: total_attacks += 4
                if test_salt_pepper: total_attacks += 4
                if test_gaussian: total_attacks += 5
                if test_scaling: total_attacks += 5
                if test_translation: total_attacks += 3
                if test_cropping: total_attacks += 3
                if test_flipping: total_attacks += 3
                if test_filtering: total_attacks += 4
                if test_format: total_attacks += 3
                
                # Run attacks
                if test_rotation:
                    for angle in [1, 5, 45]:
                        status_text.text(f"Running: Rotation {angle}°...")
                        attacked = apply_rotation_attack(watermarked_img, angle)
                        
                        # Extract from attacked
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Rotation", f"{angle}°", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Rotation", f"{angle}°", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed to extract from Rotation {angle}°: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                if test_jpeg:
                    for quality in [90, 70, 30, 10]:
                        status_text.text(f"Running: JPEG Compression Q={quality}...")
                        attacked = apply_jpeg_compression(watermarked_img, quality)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("JPEG", f"Q={quality}", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("JPEG", f"Q={quality}", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed to extract from JPEG Q={quality}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Salt & Pepper Noise
                if test_salt_pepper:
                    for density in [0.001, 0.005, 0.01, 0.02]:
                        status_text.text(f"Running: Salt & Pepper Noise d={density}...")
                        attacked = apply_salt_pepper_noise(watermarked_img, density)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Salt&Pepper", f"d={density}", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Salt&Pepper", f"d={density}", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Salt&Pepper d={density}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Gaussian Noise
                if test_gaussian:
                    for var in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
                        status_text.text(f"Running: Gaussian Noise v={var}...")
                        attacked = apply_gaussian_noise(watermarked_img, var)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Gaussian", f"v={var}", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Gaussian", f"v={var}", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Gaussian v={var}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Scaling
                if test_scaling:
                    for scale in [0.5, 0.75, 1.25, 1.5, 2.0]:
                        status_text.text(f"Running: Scaling {scale}x...")
                        attacked = apply_scaling_attack(watermarked_img, scale)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Scaling", f"{scale}x", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Scaling", f"{scale}x", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Scaling {scale}x: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Translation
                if test_translation:
                    for shift in [(10, 10), (20, 20), (50, 50)]:
                        status_text.text(f"Running: Translation {shift}...")
                        attacked = apply_translation_attack(watermarked_img, shift[0], shift[1])
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Translation", f"{shift}", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Translation", f"{shift}", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Translation {shift}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Cropping
                if test_cropping:
                    for pct in [10, 20, 25]:
                        status_text.text(f"Running: Cropping {pct}%...")
                        attacked = apply_cropping_attack(watermarked_img, pct)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Cropping", f"{pct}%", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Cropping", f"{pct}%", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Cropping {pct}%: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Flipping
                if test_flipping:
                    for flip_type in ['horizontal', 'vertical', 'both']:
                        status_text.text(f"Running: Flipping {flip_type}...")
                        attacked = apply_flipping_attack(watermarked_img, flip_type)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Flipping", flip_type, attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Flipping", flip_type, attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Flipping {flip_type}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Filtering
                if test_filtering:
                    for kernel in [3, 5, 7, 9]:
                        status_text.text(f"Running: Filtering k={kernel}...")
                        attacked = apply_filtering_attack(watermarked_img, kernel)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Filtering", f"k={kernel}", attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Filtering", f"k={kernel}", attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Filtering k={kernel}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                # Format Conversion
                if test_format:
                    for fmt in ['jpg', 'png', 'webp']:
                        status_text.text(f"Running: Format {fmt.upper()}...")
                        attacked = apply_format_conversion(watermarked_img, fmt)
                        
                        try:
                            if method == "DCT+DWT":
                                wm = WatermarkerDCTDWT(alpha=alpha_param)
                                extracted = wm.extract(attacked, dct_map, shape)
                            elif method == "Pure DWT":
                                wm = WatermarkerDWT(alpha=alpha_param)
                                extracted = wm.extract(original_img, attacked)
                            elif method == "DWT+SVD":
                                wm = WatermarkerDWTSVD(alpha=alpha_param)
                                extracted = wm.extract(attacked, original_img, ref_watermark)
                                nc = calculate_nc(ref_watermark, extracted)
                                ber, _, _ = calculate_ber(ref_watermark, extracted)
                            else:
                                wm = WatermarkerDCTLinear(alpha=alpha_param, block_size=block_size)
                                cover_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
                                img_ycrcb = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2YCrCb)
                                Y = img_ycrcb[:, :, 0]
                                h, w = Y.shape[:2]
                                num_blocks_h = h // block_size
                                num_blocks_w = w // block_size
                                total_bits = num_blocks_h * num_blocks_w
                                max_size = int(np.sqrt(total_bits))
                                wm_dims = (max_size, max_size)
                                num_bits = max_size * max_size
                                extracted = wm.extract(cover_rgb, attacked_rgb, wm_dims, num_bits)
                            
                            psnr = calculate_psnr(watermarked_img, attacked)
                            if method == "DWT+SVD":
                                attack_results.append(("Format", fmt.upper(), attacked, extracted, psnr, nc, ber))
                            else:
                                attack_results.append(("Format", fmt.upper(), attacked, extracted, psnr, None, None))
                        except Exception as e:
                            st.warning(f"Failed Format {fmt}: {str(e)}")
                        
                        current_attack += 1
                        progress_bar.progress(current_attack / total_attacks)
                
                progress_bar.progress(1.0)
                status_text.text("✅ All attacks completed!")
                
                # Display results
                st.success(f"🎉 Completed {len(attack_results)} attacks successfully!")
                
                st.subheader("Attack Results Summary")
                
                # Create summary table
                if method == "DWT+SVD":
                    st.markdown("| Attack Type | Parameter | PSNR (dB) | NC | BER (%) |")
                    st.markdown("|------------|-----------|-----------|-----|---------|")
                    for attack_type, param, _, _, psnr, nc, ber in attack_results:
                        st.markdown(f"| {attack_type} | {param} | {psnr:.2f} | {nc:.4f} | {ber:.2f} |")
                else:
                    st.markdown("| Attack Type | Parameter | PSNR (dB) |")
                    st.markdown("|------------|-----------|-----------|")
                    for attack_type, param, _, _, psnr, _, _ in attack_results:
                        st.markdown(f"| {attack_type} | {param} | {psnr:.2f} |")
                
                # Display images in grid
                st.subheader("Visual Results")
                cols_per_row = 3
                for i in range(0, len(attack_results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(attack_results):
                            attack_type, param, attacked_img, extracted_img, psnr, nc, ber = attack_results[i + j]
                            with cols[j]:
                                st.markdown(f"**{attack_type} - {param}**")
                                st.image(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB), caption="Attacked", width="stretch")
                                st.image(extracted_img, caption="Extracted", width="stretch", clamp=True)
                                if method == "DWT+SVD":
                                    st.caption(f"PSNR: {psnr:.2f}dB | NC: {nc:.4f} | BER: {ber:.2f}%")
                                else:
                                    st.caption(f"PSNR: {psnr:.2f}dB")

# =============================================================================
# Tab 4: Admin Panel (admins only)
# =============================================================================
if _is_admin and tab4 is not None:
    with tab4:
        st.header("🛡️ Admin Panel")

        admin_tab1, admin_tab2 = st.tabs(["👥 Manage Users", "📜 Activity Logs"])

        # --- Manage Users -------------------------------------------------
        with admin_tab1:
            st.subheader("Existing Users")
            users = auth.get_all_users()
            if users:
                st.dataframe(users, width="stretch", hide_index=True)
            else:
                st.info("No users found.")

            st.markdown("---")
            st.subheader("Add a New User")
            with st.form("admin_add_user_form"):
                add_username = st.text_input("Username", key="admin_add_username")
                add_password = st.text_input("Password", type="password", key="admin_add_password")
                add_role = st.selectbox("Role", ["user", "admin"], key="admin_add_role")
                if st.form_submit_button("➕ Create User", type="primary"):
                    ok, msg = auth.create_user(add_username, add_password, role=add_role)
                    if ok:
                        auth.log_activity(
                            st.session_state["username"],
                            "admin_create_user",
                            f"created={add_username}, role={add_role}"
                        )
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("---")
            st.subheader("Reset a User's Password")
            usernames = [u["username"] for u in users]
            if usernames:
                with st.form("admin_reset_password_form"):
                    reset_target = st.selectbox("User", usernames, key="admin_reset_target")
                    reset_new_pw = st.text_input("New password", type="password", key="admin_reset_pw")
                    if st.form_submit_button("🔄 Reset Password"):
                        ok, msg = auth.reset_password(reset_target, reset_new_pw)
                        if ok:
                            auth.log_activity(
                                st.session_state["username"],
                                "admin_reset_password",
                                f"target={reset_target}"
                            )
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")

            st.markdown("---")
            st.subheader("Remove a User")
            if usernames:
                with st.form("admin_delete_user_form"):
                    delete_target = st.selectbox("User", usernames, key="admin_delete_target")
                    if st.form_submit_button("🗑️ Delete User", type="primary"):
                        ok, msg = auth.delete_user(delete_target)
                        if ok:
                            auth.log_activity(
                                st.session_state["username"],
                                "admin_delete_user",
                                f"target={delete_target}"
                            )
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

        # --- Activity Logs --------------------------------------------------
        with admin_tab2:
            st.subheader("Recent Activity")
            filter_user = st.selectbox(
                "Filter by user (optional)",
                ["All users"] + [u["username"] for u in auth.get_all_users()],
                key="admin_log_filter"
            )
            logs = auth.get_logs(
                limit=300,
                username_filter=None if filter_user == "All users" else filter_user
            )
            if logs:
                st.dataframe(
                    [{k: v for k, v in row.items() if k != "id"} for row in logs],
                    width="stretch",
                    hide_index=True
                )
            else:
                st.info("No activity recorded yet.")

# Footer
st.markdown("---")
st.markdown("### 📚 About")
st.info("""
This application implements four digital watermarking techniques with **non-blind extraction**:

- **DCT+DWT** (Non-blind): Embeds in LL sub-band using 2x2 DCT blocks
  - ✅ Text → Image → Embed
  - 🔑 Extraction requires: Watermarked image + Embedding keys

- **Pure DWT** (Non-blind): Embeds in LH sub-band (horizontal detail)
  - ✅ Text → Image → Embed
  - 📷 Extraction requires: Watermarked image + Original cover image

- **DWT+SVD** (Non-blind): Medical imaging with timestamp
  - ✅ Text → Image with timestamp → Embed
  - 🏥 Extraction requires: Watermarked image + Original cover image + Original watermark

- **DCT+Linear Modulation** (Non-blind): Mid-frequency DCT with linear modulation
  - ✅ Text/Image → Embed
  - 📷 Extraction requires: Watermarked image + Original cover image

**All methods are non-blind**, meaning they require additional information beyond just the watermarked image for extraction.
""")
