// Main JavaScript functionality for the AI Applications website

// Initialize Lucide icons
lucide.createIcons();

// Mobile menu toggle
document.getElementById('mobile-menu-btn').addEventListener('click', function() {
    const mobileMenu = document.getElementById('mobile-menu');
    const icon = this.querySelector('i');
    
    if (mobileMenu.classList.contains('hidden')) {
        mobileMenu.classList.remove('hidden');
        icon.setAttribute('data-lucide', 'x');
    } else {
        mobileMenu.classList.add('hidden');
        icon.setAttribute('data-lucide', 'menu');
    }
    lucide.createIcons();
});

// Security banner close functionality
function closeBanner() {
    document.getElementById('security-banner').style.display = 'none';
}

// Auto-hide security banner after 10 seconds
setTimeout(closeBanner, 10000);

// Smooth scrolling for navigation links
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

// Scroll to top function
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Render applications grid
function renderApplications() {
    const grid = document.getElementById('applications-grid');
    
    applications.forEach(app => {
        const appCard = document.createElement('div');
        appCard.className = 'bg-white rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow cursor-pointer';
        appCard.onclick = () => openModal(app);
        
        const categoryColor = categoryColors[app.category] || 'bg-gray-100 text-gray-800';
        
        appCard.innerHTML = `
            <div class="p-6">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex-1">
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full ${categoryColor} mb-2">
                            ${app.category}
                        </span>
                        <h3 class="text-lg font-semibold text-gray-800 hover:text-blue-600 transition-colors">
                            ${app.area}
                        </h3>
                    </div>
                    <i data-lucide="external-link" class="h-4 w-4 text-gray-400"></i>
                </div>
                <div class="space-y-2 text-sm">
                    <div>
                        <span class="font-medium text-gray-700">Examples:</span>
                        <span class="text-gray-600">${app.example}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Description:</span>
                        <span class="text-gray-600">${app.description}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Sources:</span>
                        <span class="text-blue-600 text-xs">${app.sources}</span>
                    </div>
                </div>
                <div class="mt-4 flex items-center justify-between">
                    <span class="inline-block px-2 py-1 text-xs font-medium rounded-full ${difficultyColors[app.difficulty]}">
                        ${app.difficulty}
                    </span>
                    <span class="text-xs text-gray-500">${app.implementationTime}</span>
                </div>
            </div>
        `;
        
        grid.appendChild(appCard);
    });
    
    // Reinitialize Lucide icons
    lucide.createIcons();
}

// Open application detail modal
function openModal(app) {
    const modal = document.getElementById('detail-modal');
    const title = document.getElementById('modal-title');
    const content = document.getElementById('modal-content');
    
    title.textContent = app.area;
    
    const categoryColor = categoryColors[app.category] || 'bg-gray-100 text-gray-800';
    const difficultyColor = difficultyColors[app.difficulty];
    
    content.innerHTML = `
        <div class="mb-6">
            <div class="flex items-center gap-2 mb-4">
                <span class="inline-block px-3 py-1 text-sm font-medium rounded-full ${categoryColor}">
                    ${app.category}
                </span>
                <span class="inline-block px-3 py-1 text-sm font-medium rounded-full ${difficultyColor}">
                    ${app.difficulty}
                </span>
            </div>
            <p class="text-lg text-gray-600 leading-relaxed">${app.description}</p>
        </div>

        <div class="grid lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2 space-y-8">
                <!-- Overview -->
                <div class="bg-gray-50 rounded-lg p-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i data-lucide="book-open" class="h-5 w-5 mr-2"></i>
                        Overview
                    </h3>
                    <div class="space-y-4">
                        <div>
                            <h4 class="font-semibold text-gray-700 mb-2">Practical Examples</h4>
                            <p class="text-gray-600">${app.example}</p>
                        </div>
                        <hr class="border-gray-300">
                        <div>
                            <h4 class="font-semibold text-gray-700 mb-2">Technical Description</h4>
                            <p class="text-gray-600">${app.detailedDescription}</p>
                        </div>
                    </div>
                </div>

                <!-- Implementation Guide -->
                <div class="bg-gray-50 rounded-lg p-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i data-lucide="settings" class="h-5 w-5 mr-2"></i>
                        Implementation Guide
                    </h3>
                    <div class="space-y-6">
                        <div>
                            <h4 class="font-semibold text-gray-700 mb-3">Required Hardware</h4>
                            <ul class="list-disc list-inside space-y-1 text-gray-600">
                                ${app.hardware.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                        
                        <div>
                            <h4 class="font-semibold text-gray-700 mb-3">Required Software</h4>
                            <ul class="list-disc list-inside space-y-1 text-gray-600">
                                ${app.software.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>

                        <div>
                            <h4 class="font-semibold text-gray-700 mb-3">Step-by-Step Setup</h4>
                            <ol class="list-decimal list-inside space-y-2 text-gray-600">
                                ${app.setup.map(step => `<li class="leading-relaxed">${step}</li>`).join('')}
                            </ol>
                        </div>
                    </div>
                </div>

                ${app.codeExample ? `
                <!-- Code Examples -->
                <div class="bg-gray-50 rounded-lg p-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i data-lucide="code" class="h-5 w-5 mr-2"></i>
                        Code Example
                    </h3>
                    <pre class="bg-gray-100 p-4 rounded-lg overflow-x-auto text-sm"><code>${app.codeExample}</code></pre>
                </div>
                ` : ''}

                <!-- Troubleshooting -->
                <div class="bg-gray-50 rounded-lg p-6">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i data-lucide="alert-circle" class="h-5 w-5 mr-2"></i>
                        Troubleshooting
                    </h3>
                    <div class="space-y-4">
                        ${app.troubleshooting.map(item => `
                            <div>
                                <h4 class="font-medium text-gray-700 mb-1">${item.issue}</h4>
                                <p class="text-gray-600 text-sm">${item.solution}</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>

            <!-- Sidebar -->
            <div class="space-y-6">
                <!-- Quick Info -->
                <div class="bg-white border border-gray-200 rounded-lg p-6">
                    <h4 class="text-lg font-semibold text-gray-800 mb-4">Quick Information</h4>
                    <div class="space-y-4">
                        <div>
                            <h5 class="font-semibold text-gray-700 mb-1">Difficulty Level</h5>
                            <span class="inline-block px-2 py-1 text-sm font-medium rounded-full ${difficultyColor}">
                                ${app.difficulty}
                            </span>
                        </div>
                        <div>
                            <h5 class="font-semibold text-gray-700 mb-1">Implementation Time</h5>
                            <p class="text-gray-600">${app.implementationTime}</p>
                        </div>
                        <div>
                            <h5 class="font-semibold text-gray-700 mb-1">Cost Estimate</h5>
                            <p class="text-gray-600">${app.cost}</p>
                        </div>
                    </div>
                </div>

                <!-- Sources -->
                <div class="bg-white border border-gray-200 rounded-lg p-6">
                    <h4 class="text-lg font-semibold text-gray-800 mb-4">GitHub Resources & References</h4>
                    <div class="space-y-2">
                        ${app.sourceLinks.map(source => `
                            <a href="${source.url}" target="_blank" rel="noopener noreferrer" 
                               class="flex items-center text-blue-600 hover:text-blue-800 text-sm hover:underline">
                                <i data-lucide="external-link" class="h-3 w-3 mr-2 flex-shrink-0"></i>
                                <span class="truncate">${source.title}</span>
                            </a>
                        `).join('')}
                    </div>
                </div>

                <!-- Related Applications -->
                <div class="bg-white border border-gray-200 rounded-lg p-6">
                    <h4 class="text-lg font-semibold text-gray-800 mb-4">Related Applications</h4>
                    <div class="space-y-2">
                        ${applications
                            .filter(relatedApp => relatedApp.category === app.category && relatedApp.slug !== app.slug)
                            .slice(0, 3)
                            .map(relatedApp => `
                                <div class="text-sm text-blue-600 hover:text-blue-800 cursor-pointer hover:underline"
                                     onclick="openModal(applications.find(a => a.slug === '${relatedApp.slug}'))">
                                    ${relatedApp.area}
                                </div>
                            `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    
    // Reinitialize Lucide icons
    lucide.createIcons();
}

// Close modal
function closeModal() {
    const modal = document.getElementById('detail-modal');
    modal.classList.add('hidden');
    document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
document.getElementById('detail-modal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

// Escape key to close modal
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    renderApplications();
    
    // Initialize Lucide icons
    lucide.createIcons();
    
    // Add intersection observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all application cards for animation
    setTimeout(() => {
        document.querySelectorAll('#applications-grid > div').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            observer.observe(card);
        });
    }, 100);
});