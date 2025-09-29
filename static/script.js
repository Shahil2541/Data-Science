document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const predictBtn = document.getElementById("predictBtn");
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
    
    // Show loading state
    predictBtn.disabled = true;
    predictBtn.textContent = "Processing...";
    loading.style.display = 'block';
    result.classList.remove('show');
    
    try {
        const formData = new FormData(e.target);
        let data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value);
            // Validate input range
            if (data[key] < 0 || data[key] > 100) {
                throw new Error(`Please enter values between 0 and 100 for ${key.replace('_', ' ')}`);
            }
        });

        const res = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const response = await res.json();
        
        if (response.status === "success") {
            const prediction = response.prediction;
            let riskLevel = "";
            let riskClass = "";
            
            // Determine risk level
            if (prediction <= 0.50) {
                riskLevel = "Low AI Impact";
                riskClass = "risk-low";
            } else if (prediction <= 0.75) {
                riskLevel = "Medium AI Impact";
                riskClass = "risk-medium";
            } else {
                riskLevel = "High AI Impact";
                riskClass = "risk-high";
            }
            
            result.innerHTML = `
                <h3>Prediction Result</h3>
                <div style="font-size: 2rem; margin: 10px 0;">${prediction}</div>
                <div class="risk-level ${riskClass}">${riskLevel}</div>
                <p style="margin-top: 15px; font-size: 0.9rem;">
                    This job role has a ${prediction*100}% probability of being significantly impacted by AI in the near future.
                </p>
            `;
            result.classList.add('show');
        } else {
            throw new Error(response.error || "Prediction failed");
        }
        
    } catch (error) {
        result.innerHTML = `
            <h3>Error</h3>
            <p>${error.message}</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">
                Please check your inputs and try again.
            </p>
        `;
        result.classList.add('show');
    } finally {
        // Reset loading state
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict AI Impact";
        loading.style.display = 'none';
    }
});

// Add input validation
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        if (value < 0) e.target.value = 0;
        if (value > 100) e.target.value = 100;
    });
});

// Add some sample data filling for demonstration
document.addEventListener('DOMContentLoaded', () => {
    // This is just for demo purposes - remove in production
    const sampleData = {
        'Automation_Potential': 75,
        'Skill_Level': 60,
        'Experience_Years': 45,
        'Education_Level': 70,
        'Technical_Skills': 55,
        'Creative_Thinking': 80,
        'Social_Intelligence': 65,
        'Physical_Dexterity': 40
    };
    
    // Uncomment the line below to auto-fill sample data for testing
    // Object.keys(sampleData).forEach(key => { if(document.getElementById(key)) document.getElementById(key).value = sampleData[key]; });
});
// Contact Form Handling
document.getElementById("contactForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const submitBtn = e.target.querySelector('.submit-btn');
    const formSuccess = document.getElementById('formSuccess');
    
    // Show loading state
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = "Sending...";
    formSuccess.classList.remove('show');
    
    try {
        // Simulate form submission (replace with actual backend endpoint)
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Show success message
        formSuccess.classList.add('show');
        e.target.reset();
        
        // Hide success message after 5 seconds
        setTimeout(() => {
            formSuccess.classList.remove('show');
        }, 5000);
        
    } catch (error) {
        alert('There was an error sending your message. Please try again.');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
});

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});