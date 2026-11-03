/**
 * StitchSense Main Client-Side Controller
 * Theme: Nude Pink, Cream & Soft Neutrals
 * Handles drag-and-drop, image previewing, client-side validation, and UI state.
 */

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const previewBox = document.getElementById("file-preview-box");
    const previewImg = document.getElementById("preview-img");
    const previewFilename = document.getElementById("preview-filename");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const btnSubmit = document.getElementById("btn-submit");
    const btnRemove = document.getElementById("btn-remove-file");
    const spinner = document.getElementById("spinner");
    const uploadForm = document.getElementById("upload-form");

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB

    if (dropzone && fileInput) {
        // Handle Drag & Drop highlight effects
        ["dragenter", "dragover"].forEach(evtName => {
            dropzone.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evtName => {
            dropzone.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        // Handle dropped files
        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                displayPreview(files[0]);
            }
        });

        // Handle manual file selection
        fileInput.addEventListener("change", (e) => {
            if (fileInput.files && fileInput.files.length > 0) {
                displayPreview(fileInput.files[0]);
            }
        });
    }

    function displayPreview(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, WEBP).");
            resetUpload();
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            alert("File size exceeds 16MB limit. Please choose a smaller photo.");
            resetUpload();
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) previewImg.src = e.target.result;
            if (previewFilename) {
                previewFilename.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
            }
            if (dropzonePrompt) dropzonePrompt.classList.add("d-none");
            if (previewBox) previewBox.classList.remove("d-none");
            if (btnSubmit) btnSubmit.removeAttribute("disabled");
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        if (fileInput) fileInput.value = "";
        if (previewImg) previewImg.src = "";
        if (dropzonePrompt) dropzonePrompt.classList.remove("d-none");
        if (previewBox) previewBox.classList.add("d-none");
        if (btnSubmit) btnSubmit.setAttribute("disabled", "true");
    }

    if (btnRemove) {
        btnRemove.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            resetUpload();
        });
    }

    // Submit loading state
    if (uploadForm && btnSubmit) {
        uploadForm.addEventListener("submit", () => {
            btnSubmit.setAttribute("disabled", "true");
            if (spinner) spinner.classList.remove("d-none");
            const btnText = btnSubmit.querySelector(".btn-text span");
            if (btnText) btnText.textContent = "Analyzing Seam Defects...";
        });
    }
});
