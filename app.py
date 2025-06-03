import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


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
    # 1. Resize ke ukuran input final model menggunakan PIL
    image_pil_resized = image_pil.resize((target_width, target_height))

    # 2. Konversi PIL Image ke NumPy array
    image_np = np.array(image_pil_resized)

    # Pastikan format channel sesuai (misalnya, buang alpha channel jika ada)
    if image_np.shape[-1] == 4: # RGBA
        image_np = image_np[:, :, :3] # Ambil RGB saja
    
    # Jika model dilatih dengan gambar grayscale tapi input PIL berwarna, konversi di sini.
    # Atau sebaliknya, jika model RGB tapi input PIL grayscale.
    # Contoh: Jika model RGB dan input mungkin grayscale
    if image_np.ndim == 2: # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB) # Konversi ke RGB

    # 3. Terapkan enhancements (noise reduction, histogram equalization)
    # Fungsi enhance_single_image menerima NumPy array (misalnya uint8) dan mengembalikan float32 NumPy
    image_enhanced_np = enhance_single_image(image_np) # image_np harusnya sudah uint8 atau float 0-255

    # 4. Konversi NumPy array hasil enhancement ke TensorFlow tensor
    image_enhanced_tf = tf.convert_to_tensor(image_enhanced_np, dtype=tf.float32)

    # 5. Terapkan preprocessing spesifik MobileNet
    # Fungsi mobilenet_preprocess_single_image menerima tensor float32
    image_model_ready_tf = mobilenet_preprocess_single_image(image_enhanced_tf)

    # 6. Tambahkan batch dimension
    image_batch = tf.expand_dims(image_model_ready_tf, axis=0)

    return image_batch


st.title("Aerial Photo Prediction")
# ... (kode file uploader Anda) ...
uploaded_file = st.file_uploader("Choose an aerial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image_pil = Image.open(uploaded_file)
    display_width = 300
    st.image(image_pil, caption="Uploaded Image", width=display_width)

    if st.button("Predict"):
        with st.spinner("Processing and predicting..."):
            try:
                # Gunakan fungsi preprocessing yang sudah lengkap
                processed_image_batch = preprocess_image_for_prediction(image_pil, IMG_HEIGHT, IMG_WIDTH)

                # Lakukan prediksi
                # Pastikan 'model' adalah instance model Keras yang sudah di-load
                if hasattr(model, 'predict'): # Cek sederhana apakah 'model' bisa melakukan prediksi
                    predictions = model.predict(processed_image_batch)

                    # Tampilkan hasil prediksi (sesuaikan dengan output model Anda)
                    st.subheader("Prediction Results")
                    # Contoh untuk klasifikasi:
                    if CLASS_NAMES and len(predictions[0]) == len(CLASS_NAMES):
                        predicted_class_index = np.argmax(predictions[0])
                        predicted_class_name = CLASS_NAMES[predicted_class_index]
                        confidence = np.max(predictions[0]) * 100
                        st.write(f"**Predicted Class:** {predicted_class_name}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                    else:
                        st.write("Raw Predictions:", predictions)
                else:
                    st.error("Model object does not seem to be a Keras model with a 'predict' method.")

            except Exception as e:
                st.error(f"An error occurred during preprocessing or prediction: {e}")
                import traceback
                st.error(traceback.format_exc()) # Tampilkan traceback untuk debug

elif model is None:
    st.warning("Model could not be loaded. Please check the model path and file.")

st.sidebar.info(
    "Ini adalah demo model untuk prediksi Citra berbagai kelas yang diambil dari udara"
)