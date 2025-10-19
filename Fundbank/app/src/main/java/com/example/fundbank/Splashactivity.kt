package com.example.fundbank

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.OvershootInterpolator
import androidx.appcompat.app.AppCompatActivity
import com.example.fundbank.databinding.ActivitySplashactivityBinding
import com.google.firebase.auth.FirebaseAuth

class SplashActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySplashactivityBinding
    private lateinit var auth: FirebaseAuth
    private var navigationHandled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySplashactivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        auth = FirebaseAuth.getInstance()

        // Start splash animation
        startSplashAnimation()

        // Wait for animation to complete, then check authentication
        Handler(Looper.getMainLooper()).postDelayed({
            checkUserAuthentication()
        }, 2500)
    }

    private fun startSplashAnimation() {
        val scaleXAnimator = ObjectAnimator.ofFloat(binding.ivLogo, "scaleX", 0f, 1f).apply {
            duration = 800
            interpolator = OvershootInterpolator(1.2f)
        }

        val scaleYAnimator = ObjectAnimator.ofFloat(binding.ivLogo, "scaleY", 0f, 1f).apply {
            duration = 800
            interpolator = OvershootInterpolator(1.2f)
        }

        val fadeInAnimator = ObjectAnimator.ofFloat(binding.ivLogo, "alpha", 0f, 1f).apply {
            duration = 600
            interpolator = AccelerateDecelerateInterpolator()
        }

        val progressFadeIn = ObjectAnimator.ofFloat(binding.progressBar, "alpha", 0f, 1f).apply {
            duration = 400
            startDelay = 600
            interpolator = AccelerateDecelerateInterpolator()
        }

        val animatorSet = AnimatorSet()
        animatorSet.playTogether(scaleXAnimator, scaleYAnimator, fadeInAnimator, progressFadeIn)
        animatorSet.start()
    }

    private fun checkUserAuthentication() {
        if (navigationHandled) return
        navigationHandled = true

        val currentUser = auth.currentUser
        val sharedPref = getSharedPreferences("FundBankPrefs", MODE_PRIVATE)
        val isFirstTime = sharedPref.getBoolean("isFirstTime", true)

        // FIXED: Check login status FIRST, then check first time
        when {
            currentUser != null -> {
                // User is logged in - ALWAYS go to main activity
                navigateToActivity(MainActivity::class.java)
            }
            isFirstTime -> {
                // Not logged in AND first time - show onboarding
                navigateToActivity(OnboardingActivity::class.java)
            }
            else -> {
                // Not logged in AND not first time - go to login
                navigateToActivity(LoginActivity::class.java)
            }
        }
    }

    private fun navigateToActivity(activityClass: Class<*>) {
        val intent = Intent(this, activityClass)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }
}