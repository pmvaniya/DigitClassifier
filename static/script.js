document.addEventListener("DOMContentLoaded", () => {
	const canvas = document.getElementById("myCanvas");
	const ctx = canvas.getContext("2d");
	const predictButton = document.getElementById("predict-btn");

	let isDrawing = false;

	canvas.addEventListener("mousedown", (e) => {
		isDrawing = true;
		draw(e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop);
	});

	canvas.addEventListener("mousemove", (e) => {
		if (isDrawing) {
			draw(e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop, true);
		}
	});

	canvas.addEventListener("mouseup", () => {
		isDrawing = false;
	});

	canvas.addEventListener("mouseleave", () => {
		isDrawing = false;
	});

	function draw(x, y, isDrawing) {
		if (!isDrawing) {
			ctx.beginPath();
		}
		ctx.lineWidth = 36;
		ctx.lineCap = "round";
		ctx.strokeStyle = "#000";
		ctx.lineTo(x, y);
		ctx.stroke();
	}

	predictButton.addEventListener("click", () => {
		const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		const pixelData = imageData.data;
		const colorsArray = [];

		for (let i = 0; i < pixelData.length; i += 4) {
			const red = pixelData[i];
			const green = pixelData[i + 1];
			const blue = pixelData[i + 2];
			const alpha = pixelData[i + 3];
			// colorsArray.push([red, green, blue, alpha]);
			colorsArray.push(alpha);
		}

		if (colorsArray.every(value => value === 0)) {
			identifyImage();
		} else {
			identifyCanvas(colorsArray);
			ctx.clearRect(0, 0, canvas.width, canvas.height);
		}
	});
});

function identifyImage() {
	const fileInput = document.getElementById('fileInput');
	const file = fileInput.files[0];
	const formData = new FormData();
	formData.append('image', file);

	const xhr = new XMLHttpRequest();
	xhr.open('POST', '/identifyImage', true);
	xhr.onload = function () {
		if (xhr.status === 200) {
			const response = JSON.parse(xhr.responseText);
			document.getElementById("predict-ans").innerHTML = response["message"];

			document.getElementById('fileInput').value = '';
			document.getElementById("imageSpan").innerHTML = '<b>OR</b> Upload an Image Here <i class="bi bi-image"></i>';
			document.getElementById("imageSpan").style.color = "";
		}
	};
	xhr.send(formData);
}

function identifyCanvas(colorsArray) {
	const xhr = new XMLHttpRequest();

	xhr.open("POST", "/identifyCanvas", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
			const response = JSON.parse(xhr.responseText);
			document.getElementById("predict-ans").innerHTML = response["message"];
		}
	};
	xhr.send(JSON.stringify({ data: colorsArray }));
}

document.getElementById('fileInput').addEventListener('change', function () {
	var uploadedFile = this.files[0];
	document.getElementById("imageSpan").innerHTML = "File uploaded: " + uploadedFile.name;
	document.getElementById("imageSpan").style.color = "green";
});