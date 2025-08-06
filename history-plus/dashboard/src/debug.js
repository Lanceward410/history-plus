// Enhanced debug dashboard toggle with multiple event listener approaches
function toggleDebugDashboard() {
    const collapsed = document.getElementById('debugCollapsed');
    const expanded = document.getElementById('debugDashboard');
    
    console.log('toggleDebugDashboard called');
    console.log('Collapsed:', collapsed ? 'found' : 'NOT FOUND');
    console.log('Expanded:', expanded ? 'found' : 'NOT FOUND');
    
    if (!collapsed || !expanded) {
        console.error('Debug dashboard elements not found!');
        return;
    }
    
    // Check current state
    const isExpanded = expanded.style.display === 'block';
    console.log('Current state - isExpanded:', isExpanded);
    
    if (isExpanded) {
        // Collapse
        console.log('Collapsing debug dashboard');
        collapsed.style.display = 'block';
        expanded.style.display = 'none';
    } else {
        // Expand
        console.log('Expanding debug dashboard');
        collapsed.style.display = 'none';
        expanded.style.display = 'block';
        
        // Start monitoring if debug dashboard is available
        setTimeout(() => {
            if (window.dashboard && window.dashboard.debugDashboard) {
                window.dashboard.debugDashboard.refreshAllMetrics();
            }
        }, 100);
    }
}

// Make function globally available immediately
window.toggleDebugDashboard = toggleDebugDashboard;

// Set up multiple event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Setting up debug dashboard event listeners...');
    
    const collapsed = document.getElementById('debugCollapsed');
    const expanded = document.getElementById('debugDashboard');
    
    if (collapsed) {
        console.log(' Adding click listeners to debug collapsed element');
        
        // Remove any existing onclick to avoid conflicts
        collapsed.removeAttribute('onclick');
        
        // Add modern event listener
        collapsed.addEventListener('click', function(e) {
            console.log('Debug collapsed clicked via event listener');
            e.preventDefault();
            e.stopPropagation();
            toggleDebugDashboard();
        });
        
        // Also add to the content div specifically
        const content = collapsed.querySelector('.debug-collapsed-content');
        if (content) {
            content.removeAttribute('onclick');
            content.addEventListener('click', function(e) {
                console.log('Debug collapsed content clicked');
                e.preventDefault();
                e.stopPropagation();
                toggleDebugDashboard();
            });
        }
    } else {
        console.error(' Debug collapsed element not found!');
    }
    
    // Set up collapse button
    const collapseBtn = document.querySelector('.debug-collapse-btn');
    if (collapseBtn) {
        collapseBtn.removeAttribute('onclick');
        collapseBtn.addEventListener('click', function(e) {
            console.log('Debug collapse button clicked');
            e.preventDefault();
            e.stopPropagation();
            toggleDebugDashboard();
        });
    }
    
    console.log('Debug dashboard setup complete');
});

// Also try immediate setup in case DOM is already loaded
if (document.readyState === 'loading') {
    console.log('DOM still loading, waiting...');
} else {
    console.log('DOM already loaded, setting up immediately...');
    setTimeout(() => {
        const collapsed = document.getElementById('debugCollapsed');
        if (collapsed && !collapsed.hasAttribute('data-listeners-added')) {
            collapsed.setAttribute('data-listeners-added', 'true');
            collapsed.addEventListener('click', toggleDebugDashboard);
        }
    }, 100);
}
