async function upload() {
    const fileInput = document.getElementById("audioFile");
    const result = document.getElementById("result");
    const loader = document.getElementById("loader");

    if (!fileInput.files[0]) {
        alert("Please upload an audio file.");
        return;
    }

    result.innerText = "";
    loader.style.display = "block";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        loader.style.display = "none";

        result.innerText = `Event: ${data.event}  
Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    } catch (error) {
        loader.style.display = "none";
        result.innerText = "Error: Cannot connect to backend!";
    }
}
