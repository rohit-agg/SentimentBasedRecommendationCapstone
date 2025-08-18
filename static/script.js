document.addEventListener('DOMContentLoaded', function() {
  document.getElementById("recommendForm").addEventListener("submit", function() {
    const btn = document.getElementById("submitBtn");
    const spinner = document.getElementById("spinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";
  });
});
