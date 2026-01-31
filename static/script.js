document.addEventListener('DOMContentLoaded', () => {
    // --- NAVIGATION LOGIC ---
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page-content');

    // Settings State
    let isHighRiskAlertsEnabled = true; // Default ON per UI
    let isDarkMode = false;

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('data-page');

            // 1. Update Active State on Sidebar
            navLinks.forEach(l => {
                l.classList.remove('bg-accent-600', 'text-white', 'shadow-lg');
                l.classList.add('text-slate-400', 'hover:text-white', 'hover:bg-white/5');
            });
            link.classList.remove('text-slate-400', 'hover:text-white', 'hover:bg-white/5');
            link.classList.add('bg-accent-600', 'text-white', 'shadow-lg');

            // 2. Show Target Page
            pages.forEach(page => {
                if (page.id === `page-${targetId}`) {
                    page.classList.remove('hidden');
                    page.classList.add('block', 'animate-fade-in');
                } else {
                    page.classList.add('hidden');
                    page.classList.remove('block', 'animate-fade-in');
                }
            });
        });
    });

    // --- PREDICTION LOGIC ---
    const form = document.getElementById('predictForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const resultContainer = document.getElementById('resultContainer');
    const emptyState = document.getElementById('emptyState');
    const predictionValue = document.getElementById('predictionValue');
    const predictionText = document.getElementById('predictionText');
    const gaugeCircle = document.getElementById('gaugeCircle');
    const historyTableBody = document.getElementById('historyTableBody');
    const noHistoryRow = document.getElementById('noHistoryRow');

    // Gauge Configuration
    const radius = 88;
    const circumference = 2 * Math.PI * radius;
    gaugeCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    gaugeCircle.style.strokeDashoffset = circumference;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Loading State
        predictBtn.disabled = true;
        predictBtn.classList.add('opacity-75', 'cursor-not-allowed');
        btnText.textContent = 'Analyzing...';

        // Gather Data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value);
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (response.ok) {
                const finalValue = result.prediction;
                displayResult(finalValue);
                addToHistory(data, finalValue);
            } else {
                alert('Error: ' + (result.error || 'Unknown error occurred'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the prediction server.');
        } finally {
            // Reset Button
            predictBtn.disabled = false;
            predictBtn.classList.remove('opacity-75', 'cursor-not-allowed');
            btnText.textContent = 'Run Prediction Model';
        }
    });

    function displayResult(value) {
        // Switch Views
        emptyState.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        resultContainer.classList.add('animate-fade-in');

        // Update Text
        predictionValue.textContent = '0%';

        // Calculate Risk Level
        let riskText = '';
        let colorClass = '';

        if (value < 30) {
            riskText = 'Low Risk of Automation';
            colorClass = 'text-emerald-400';
            gaugeCircle.classList.remove('text-tech-500', 'text-amber-500', 'text-red-500');
            gaugeCircle.classList.add('text-emerald-500');
        } else if (value < 70) {
            riskText = 'Moderate Risk of Automation';
            colorClass = 'text-amber-400';
            gaugeCircle.classList.remove('text-tech-500', 'text-emerald-500', 'text-red-500');
            gaugeCircle.classList.add('text-amber-500');
        } else {
            riskText = 'High Risk of Automation';
            colorClass = 'text-red-400';
            gaugeCircle.classList.remove('text-tech-500', 'text-emerald-500', 'text-amber-500');
            gaugeCircle.classList.add('text-red-500');
        }

        predictionText.innerHTML = `<span class="${colorClass} font-bold">${riskText}</span><br><span class="text-sm text-slate-400 font-normal">Based on current job market analysis</span>`;

        // Animate Number
        let current = 0;
        const step = value / 100;
        const interval = setInterval(() => {
            current += step;
            if (current >= value) {
                current = value;
                clearInterval(interval);
            }
            predictionValue.textContent = Math.round(current) + '%';
        }, 10);

        // Animate Gauge
        const offset = circumference - (value / 100) * circumference;
        gaugeCircle.style.strokeDashoffset = offset;

        // High Risk Alert
        if (value > 75 && isHighRiskAlertsEnabled) {
            setTimeout(() => {
                alert("⚠️ HIGH RISK WARNING:\n\nAnalysis indicates a >75% probability of AI automation for this role.\nImmediate reskilling is recommended.");
            }, 1000); // Small delay to let animation start
        }
    }

    function addToHistory(inputs, score) {
        // Remove empty state row if it exists
        if (noHistoryRow) {
            noHistoryRow.style.display = 'none';
        }

        const date = new Date();
        const timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Create Badge
        let badgeClass = '';
        let statusText = '';
        if (score < 30) {
            badgeClass = 'bg-emerald-100 text-emerald-800 border-emerald-200';
            statusText = 'Safe';
        } else if (score < 70) {
            badgeClass = 'bg-amber-100 text-amber-800 border-amber-200';
            statusText = 'Moderate';
        } else {
            badgeClass = 'bg-red-100 text-red-800 border-red-200';
            statusText = 'Risk';
        }

        // Create Row
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50 transition-colors border-b border-slate-100';
        tr.innerHTML = `
            <td class="px-6 py-4 font-medium text-slate-900">${timeString}</td>
            <td class="px-6 py-4 text-slate-500 font-mono text-xs">
                ${Object.values(inputs).slice(0, 3).join(', ')}...
            </td>
            <td class="px-6 py-4 font-bold text-brand-900">${Math.round(score)}%</td>
            <td class="px-6 py-4">
                <span class="px-2 py-1 rounded-full text-xs font-semibold border ${badgeClass}">
                    ${statusText}
                </span>
            </td>
        `;

        // Prepend to table
        historyTableBody.insertBefore(tr, historyTableBody.firstChild);
    }

    // --- NOTIFICATION LOGIC ---
    const notificationBtn = document.getElementById('notificationBtn');
    const notificationMenu = document.getElementById('notificationMenu');

    if (notificationBtn && notificationMenu) {
        notificationBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            notificationMenu.classList.toggle('hidden');
        });

        // Close when clicking outside
        document.addEventListener('click', (e) => {
            if (!notificationMenu.contains(e.target) && !notificationBtn.contains(e.target)) {
                notificationMenu.classList.add('hidden');
            }
        });

        // Mark All Read
        const markReadBtn = document.getElementById('markReadBtn');
        const notificationDot = document.getElementById('notificationDot');
        const notificationBadge = document.getElementById('notificationBadge');

        if (markReadBtn) {
            markReadBtn.addEventListener('click', () => {
                // hide red dot
                if (notificationDot) notificationDot.classList.add('hidden');

                // update badge
                if (notificationBadge) {
                    notificationBadge.textContent = '0 New';
                    notificationBadge.classList.remove('bg-accent-50', 'text-accent-600');
                    notificationBadge.classList.add('bg-slate-100', 'text-slate-400');
                }

                // disable button
                markReadBtn.textContent = 'All caught up';
                markReadBtn.classList.add('opacity-50', 'cursor-default');
                markReadBtn.disabled = true;
            });
        }
    }


    // --- SETTINGS LOGIC ---
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkModeKnob = document.getElementById('darkModeKnob');
    const highRiskToggle = document.getElementById('highRiskToggle');
    const highRiskKnob = document.getElementById('highRiskKnob');
    const html = document.documentElement;

    // 1. Dark Mode
    if (darkModeToggle && darkModeKnob) {
        darkModeToggle.addEventListener('click', () => {
            isDarkMode = !isDarkMode;
            if (isDarkMode) {
                html.classList.add('dark');
                // UI: ON State
                darkModeToggle.classList.replace('bg-slate-200', 'bg-accent-600');
                darkModeKnob.classList.replace('translate-x-1', 'translate-x-6');
            } else {
                html.classList.remove('dark');
                // UI: OFF State
                darkModeToggle.classList.replace('bg-accent-600', 'bg-slate-200');
                darkModeKnob.classList.replace('translate-x-6', 'translate-x-1');
            }
        });
    }

    // 2. High Risk Alerts
    if (highRiskToggle && highRiskKnob) {
        highRiskToggle.addEventListener('click', () => {
            isHighRiskAlertsEnabled = !isHighRiskAlertsEnabled;
            if (isHighRiskAlertsEnabled) {
                // UI: ON State
                highRiskToggle.classList.replace('bg-slate-200', 'bg-accent-600');
                highRiskKnob.classList.replace('translate-x-1', 'translate-x-6');
            } else {
                // UI: OFF State
                highRiskToggle.classList.replace('bg-accent-600', 'bg-slate-200');
                highRiskKnob.classList.replace('translate-x-6', 'translate-x-1');
            }
        });
    }

    // 3. Admin Name Persistence
    const adminNameInput = document.getElementById('adminNameInput');
    if (adminNameInput) {
        // Load from storage
        const savedName = localStorage.getItem('adminName');
        if (savedName) adminNameInput.value = savedName;

        // Save on input
        adminNameInput.addEventListener('input', (e) => {
            const newName = e.target.value;
            localStorage.setItem('adminName', newName);

            // Update Sidebar Name
            const sidebarName = document.getElementById('sidebarUserName');
            if (sidebarName) sidebarName.textContent = newName || 'Admin';
        });

        // Initial Load for Sidebar
        const sidebarName = document.getElementById('sidebarUserName');
        if (savedName && sidebarName) sidebarName.textContent = savedName;
    }

    // 4. Update Login Time
    const loginTimeEl = document.getElementById('loginTime');
    if (loginTimeEl) {
        const startTime = new Date();

        setInterval(() => {
            const now = new Date();
            const diffInMinutes = Math.floor((now - startTime) / 60000);

            if (diffInMinutes < 1) {
                loginTimeEl.textContent = 'Just now';
            } else if (diffInMinutes === 1) {
                loginTimeEl.textContent = '1 min ago';
            } else {
                loginTimeEl.textContent = `${diffInMinutes} mins ago`;
            }
        }, 60000); // Check every minute
    }

    // 5. Apply Dynamic Widths (Fix for HTML Linting)
    const widthTargets = document.querySelectorAll('.width-target');
    widthTargets.forEach(el => {
        const width = el.getAttribute('data-width');
        if (width) el.style.width = width;
    });
});