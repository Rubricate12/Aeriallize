import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

st.set_page_config(layout="wide")
MODEL_PATH = 'modelfinal.h5' # Path to your .h5 model file
IMG_HEIGHT = 224             # Expected image height by your model
IMG_WIDTH = 224              # Expected image width by your model
CLASS_NAMES = ['Agriculture', 'Airport', 'Beach', 'City', 'Desert', 'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain', 'Parking', 'Port', 'Railway', 'Residential', 'River'] # Replace with your actual class names

def create_your_actual_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=len(CLASS_NAMES)):
    base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # your number of classes
    ])

    return model

# --- Image Preprocessing Function ---
def preprocess_image(image_pil, target_height, target_width):
    """
    Preprocesses a PIL Image for the model.
    Adjust this function based on your model's specific preprocessing needs.
    """
    # Resize
    image_resized = image_pil.resize((target_width, target_height))

    # Convert to NumPy array
    image_array = np.array(image_resized)

    # Ensure 3 channels (if your model expects RGB)
    if image_array.ndim == 2: # Grayscale
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4: # RGBA
        image_array = image_array[:, :, :3]

    # Add batch dimension (model expects a batch of images)
    image_batch = np.expand_dims(image_array, axis=0)

    # Normalize (IMPORTANT: This must match how your model was trained)
    # Example: Scaling to [0, 1]
    # image_batch_normalized = image_batch / 255.0
    # Example: If your model used tf.keras.applications.mobilenet.preprocess_input
    image_batch_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(image_batch.copy())

    return image_batch_normalized # Or whatever your model expects

@st.cache_resource
def load_model_with_weights(weights_path):
    try:
        # 1. Create an instance of your model architecture
        #    Ensure IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES are correctly defined globally or passed
        model_instance = create_your_actual_model(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            num_classes=len(CLASS_NAMES)
        )
        # 2. Load the weights into this architecture
        model_instance.load_weights(weights_path)
        st.success("Model architecture defined and weights loaded successfully!")
        return model_instance
    except Exception as e:
        st.error(f"Error creating model or loading weights: {e}")
        return None

model = load_model_with_weights(MODEL_PATH)

def enhance_single_image(image_np_uint8):
    """Menerapkan noise reduction dan histogram equalization pada satu gambar NumPy."""
    # Pastikan input adalah uint8 jika OpenCV memerlukannya
    if image_np_uint8.dtype != np.uint8:
        # Jika float dari PIL (0-255), konversi dulu
        if np.max(image_np_uint8) > 1.0: # Asumsi masih 0-255
             image_np_uint8 = image_np_uint8.astype(np.uint8)
        else: # Asumsi sudah 0-1, skalakan ke 0-255 lalu uint8
             image_np_uint8 = (image_np_uint8 * 255).astype(np.uint8)


    # 1. Reduce Noise (Contoh menggunakan GaussianBlur dari OpenCV)
    image_denoised = cv2.GaussianBlur(image_np_uint8, (5, 5), 0)

    # 2. Histogram Equalization
    if image_denoised.shape[-1] == 3: # RGB
        img_yuv = cv2.cvtColor(image_denoised, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        image_equalized_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    elif image_denoised.ndim == 2 or image_denoised.shape[-1] == 1: # Grayscale
        image_equalized_np = cv2.equalizeHist(image_denoised)
        if image_equalized_np.ndim == 2:
             image_equalized_np = np.expand_dims(image_equalized_np, axis=-1)
    else:
        image_equalized_np = image_denoised # Fallback

    return image_equalized_np.astype(np.float32) # Kembalikan sebagai float32 untuk TensorFlow

def mobilenet_preprocess_single_image(image_tf_float32):
    """Menerapkan preprocessing MobileNet pada satu gambar tensor float32."""
    # Fungsi ini mengharapkan gambar sudah di-resize ke ukuran input model
    # dan dalam format float32.
    # MobileNet preprocess_input akan menangani penskalaan piksel ke [-1, 1]
    return tf.keras.applications.mobilenet.preprocess_input(image_tf_float32)

# --- Fungsi Preprocessing Utama untuk Streamlit ---
def preprocess_image_for_prediction(image_pil, target_height, target_width):
    """
    Preprocesses a PIL Image for model prediction, including all necessary steps.
    """
    # 1. Resize to the final model input size using PIL
    # Using ANTIALIAS filter for better quality resizing
    image_pil_resized = image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # 2. Convert PIL Image to NumPy array
    # Ensure it's RGB. If it has Alpha, remove it.
    if image_pil_resized.mode == 'RGBA':
        image_pil_resized = image_pil_resized.convert('RGB')
    elif image_pil_resized.mode == 'L': # Grayscale
        image_pil_resized = image_pil_resized.convert('RGB') # Convert to RGB as MobileNetV2 expects 3 channels

    image_np = np.array(image_pil_resized) # This will be uint8, [0, 255]

    # 3. Apply enhancements (noise reduction, histogram equalization)
    image_enhanced_np_float32 = enhance_single_image(image_np) # image_np is uint8

    # Prepare a version for display (uint8, 0-255)
    # Clamp values to ensure they are within 0-255 before converting to uint8
    image_enhanced_for_display_np = np.clip(image_enhanced_np_float32, 0, 255).astype(np.uint8)

    # 4. Convert the enhanced NumPy array (float32, 0-255) to a TensorFlow tensor
    image_enhanced_tf = tf.convert_to_tensor(image_enhanced_np_float32, dtype=tf.float32)

    # 5. Apply MobileNetV2-specific preprocessing
    image_model_ready_tf = mobilenet_preprocess_single_image(image_enhanced_tf)

    # 6. Add batch dimension (model expects a batch of images)
    image_batch_tf = tf.expand_dims(image_model_ready_tf, axis=0)

    return image_enhanced_for_display_np, image_batch_tf


# --- Streamlit App UI ---

st.title("🛰️ Aerial Scene Classification 📸")
st.markdown("Upload an aerial image (e.g., satellite or drone footage) to classify its scene type.")


col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an aerial image...", type=["jpg", "jpeg", "png"])
    
    # Display original uploaded image
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Original Uploaded Image", use_column_width=True)

with col2:
    st.header("⚙️ Processing & Prediction")
    if uploaded_file is not None and model is not None:
        if st.button("🔍 Classify Scene", type="primary", use_container_width=True):
            with st.spinner("Analyzing image... This may take a moment."):
                try:
                    # Preprocess the image for display and prediction
                    # This function now returns two images:
                    # 1. enhanced_display_img: For showing the user the preprocessing result (uint8)
                    # 2. model_input_batch: For feeding into the model (TensorFlow batch)
                    enhanced_display_img, model_input_batch = preprocess_image_for_prediction(
                        image_pil, IMG_HEIGHT, IMG_WIDTH
                    )

                    # Display the preprocessed (enhanced) image
                    st.subheader("✨ Enhanced Image (for Model Input)")
                    st.image(enhanced_display_img, caption="Image after Noise Reduction & Equalization", use_column_width=True)

                    # Perform prediction
                    if hasattr(model, 'predict'):
                        predictions = model.predict(model_input_batch)

                        st.subheader("📊 Prediction Results")
                        if CLASS_NAMES and len(predictions[0]) == len(CLASS_NAMES):
                            predicted_class_index = np.argmax(predictions[0])
                            predicted_class_name = CLASS_NAMES[predicted_class_index]
                            confidence = np.max(predictions[0]) * 100
                            
                            st.success(f"**Predicted Scene:** {predicted_class_name}")
                            st.info(f"**Confidence:** {confidence:.2f}%")

                            # Optional: Display top N predictions
                            st.markdown("---")
                            st.write("**Top Probabilities:**")
                            # Get top 3 predictions
                            top_k_indices = np.argsort(predictions[0])[::-1][:3]
                            for i in top_k_indices:
                                st.write(f"- {CLASS_NAMES[i]}: {predictions[0][i]*100:.2f}%")
                        else:
                            st.write("Raw Predictions (output shape or class names might not match):", predictions)
                    else:
                        st.error("Model object is not a valid Keras model with a 'predict' method.")

                except Exception as e:
                    st.error(f"An error occurred during processing or prediction: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
    
    elif model is None:
        st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure the weights file exists and is compatible.")
    elif uploaded_file is None:
        st.info("☝️ Please upload an image to start.")

st.sidebar.info(
    "Ini adalah demo model untuk prediksi Citra berbagai kelas yang diambil dari udara"
)