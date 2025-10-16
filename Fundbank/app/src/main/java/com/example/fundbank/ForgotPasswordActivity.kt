package com.example.fundbank

import android.os.Bundle
import android.util.Patterns
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import com.google.firebase.auth.FirebaseAuth

class ForgotPasswordActivity : AppCompatActivity() {

    private lateinit var auth: FirebaseAuth

    private lateinit var ivBack: ImageView
    private lateinit var etEmail: TextInputEditText
    private lateinit var btnResetPassword: MaterialButton
    private lateinit var tvBackToLogin: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_forgot_password)

        auth = FirebaseAuth.getInstance()

        initializeViews()
        setupClickListeners()
    }

    private fun initializeViews() {
        ivBack = findViewById(R.id.ivBack)
        etEmail = findViewById(R.id.etEmail)
        btnResetPassword = findViewById(R.id.btnResetPassword)
        tvBackToLogin = findViewById(R.id.tvBackToLogin)
    }

    private fun setupClickListeners() {
        ivBack.setOnClickListener {
            finish()
        }

        btnResetPassword.setOnClickListener {
            sendPasswordResetEmail()
        }

        tvBackToLogin.setOnClickListener {
            finish()
        }
    }

    private fun sendPasswordResetEmail() {
        val email = etEmail.text.toString().trim()

        if (!validateInput(email)) {
            return
        }

        btnResetPassword.isEnabled = false
        btnResetPassword.text = "Sending..."

        auth.sendPasswordResetEmail(email)
            .addOnCompleteListener { task ->
                btnResetPassword.isEnabled = true
                btnResetPassword.text = "Send Reset Link"

                if (task.isSuccessful) {
                    Toast.makeText(
                        this,
                        getString(R.string.reset_email_sent),
                        Toast.LENGTH_LONG
                    ).show()
                    finish()
                } else {
                    Toast.makeText(
                        this,
                        "Failed to send reset email: ${task.exception?.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
    }

    private fun validateInput(email: String): Boolean {
        if (email.isEmpty()) {
            Toast.makeText(this, getString(R.string.error_empty_email), Toast.LENGTH_SHORT).show()
            return false
        }

        if (!Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            Toast.makeText(this, getString(R.string.error_invalid_email), Toast.LENGTH_SHORT).show()
            return false
        }

        return true
    }
}