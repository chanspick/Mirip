/**
 * MIRIP Landing Page JavaScript
 * 스크롤 애니메이션, 네비게이션, 폼 처리
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initScrollAnimations();
    initScoreAnimation();
    initForm();
    initModal();
    initSmoothScroll();
});

/**
 * Navigation
 */
function initNavigation() {
    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.querySelector('.nav-menu');

    // Scroll effect for navbar
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        // Add scrolled class when scrolled down
        if (currentScroll > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    });

    // Mobile menu toggle
    navToggle.addEventListener('click', () => {
        navToggle.classList.toggle('active');
        navMenu.classList.toggle('active');
        document.body.style.overflow = navMenu.classList.contains('active') ? 'hidden' : '';
    });

    // Close mobile menu when clicking a link
    navMenu.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
            document.body.style.overflow = '';
        });
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!navMenu.contains(e.target) && !navToggle.contains(e.target)) {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
            document.body.style.overflow = '';
        }
    });
}

/**
 * Scroll Animations (Fade-in effect)
 */
function initScrollAnimations() {
    const fadeElements = document.querySelectorAll('.fade-in');

    const observerOptions = {
        root: null,
        rootMargin: '0px 0px -100px 0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    fadeElements.forEach(el => observer.observe(el));
}

/**
 * Score Bar Animation
 */
function initScoreAnimation() {
    const scoreFills = document.querySelectorAll('.score-fill');

    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.5
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const score = entry.target.dataset.score;
                entry.target.style.width = score + '%';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    scoreFills.forEach(el => observer.observe(el));
}

/**
 * Form Handling
 */
function initForm() {
    const form = document.getElementById('registration-form');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitBtn = form.querySelector('.btn-submit');
        const originalText = submitBtn.textContent;

        // Disable button and show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = '처리 중...';

        // Get form data
        const formData = {
            name: form.name.value.trim(),
            email: form.email.value.trim(),
            userType: form.userType.value,
            timestamp: new Date().toISOString()
        };

        try {
            // Firebase Firestore integration placeholder
            // In production, replace this with actual Firebase code:
            // await firebase.firestore().collection('registrations').add(formData);

            // Simulate API call
            await simulateApiCall(formData);

            // Show success modal
            showModal();

            // Reset form
            form.reset();

        } catch (error) {
            console.error('Registration error:', error);
            alert('등록 중 오류가 발생했습니다. 다시 시도해주세요.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    });
}

/**
 * Simulate API call (replace with actual Firebase integration)
 */
function simulateApiCall(data) {
    return new Promise((resolve) => {
        console.log('Registration data:', data);
        setTimeout(resolve, 1000);
    });
}

/**
 * Modal Handling
 */
function initModal() {
    const modal = document.getElementById('success-modal');
    const closeBtn = document.getElementById('modal-close');

    if (!modal || !closeBtn) return;

    closeBtn.addEventListener('click', hideModal);

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            hideModal();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            hideModal();
        }
    });
}

function showModal() {
    const modal = document.getElementById('success-modal');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function hideModal() {
    const modal = document.getElementById('success-modal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

/**
 * Smooth Scroll for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (!targetElement) return;

            const navHeight = document.getElementById('navbar').offsetHeight;
            const targetPosition = targetElement.offsetTop - navHeight - 20;

            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        });
    });
}

/**
 * Firebase Integration Template
 * Uncomment and configure when connecting to Firebase
 */
/*
// Import Firebase modules
import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc } from 'firebase/firestore';

const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

async function saveRegistration(data) {
    try {
        const docRef = await addDoc(collection(db, 'registrations'), data);
        console.log('Document written with ID:', docRef.id);
        return docRef.id;
    } catch (error) {
        console.error('Error adding document:', error);
        throw error;
    }
}
*/
