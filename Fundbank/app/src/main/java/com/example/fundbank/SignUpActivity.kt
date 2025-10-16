package com.example.fundbank

import android.content.Intent
import android.os.Bundle
import android.util.Patterns
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.fundbank.databinding.ActivitySignUpBinding
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInAccount
import com.google.android.gms.auth.api.signin.GoogleSignInClient
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.common.api.ApiException
import com.google.android.gms.tasks.Task
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.UserProfileChangeRequest
import com.google.firebase.firestore.FirebaseFirestore

class SignUpActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySignUpBinding
    private lateinit var auth: FirebaseAuth
    private lateinit var firestore: FirebaseFirestore
    private lateinit var googleSignInClient: GoogleSignInClient

    companion object {
        private const val RC_SIGN_IN = 9001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySignUpBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize Firebase
        auth = FirebaseAuth.getInstance()
        firestore = FirebaseFirestore.getInstance()

        // Configure Google Sign In
        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(getString(R.string.default_web_client_id))
            .requestEmail()
            .build()

        googleSignInClient = GoogleSignIn.getClient(this, gso)

        setupClickListeners()
    }

    private fun setupClickListeners() {
        binding.btnSignUp.setOnClickListener {
            registerUser()
        }

        binding.btnGoogleSignUp.setOnClickListener {
            signUpWithGoogle()
        }

        binding.tvSignIn.setOnClickListener {
            finish()
        }
    }

    private fun registerUser() {
        val fullName = binding.etFullName.text.toString().trim()
        val email = binding.etEmail.text.toString().trim()
        val password = binding.etPassword.text.toString().trim()
        val confirmPassword = binding.etConfirmPassword.text.toString().trim()

        if (!validateInput(fullName, email, password, confirmPassword)) {
            return
        }

        binding.btnSignUp.isEnabled = false
        binding.btnSignUp.text = "Creating Account..."

        auth.createUserWithEmailAndPassword(email, password)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser

                    val profileUpdates = UserProfileChangeRequest.Builder()
                        .setDisplayName(fullName)
                        .build()

                    user?.updateProfile(profileUpdates)
                        ?.addOnCompleteListener { updateTask ->
                            if (updateTask.isSuccessful) {
                                // Save user data to Firestore
                                user.let {
                                    saveUserDataToFirestore(it.uid, fullName, email)
                                }
                            } else {
                                binding.btnSignUp.isEnabled = true
                                binding.btnSignUp.text = "Create Account"
                                Toast.makeText(
                                    this,
                                    "Failed to update profile: ${updateTask.exception?.message}",
                                    Toast.LENGTH_SHORT
                                ).show()
                            }
                        }
                } else {
                    binding.btnSignUp.isEnabled = true
                    binding.btnSignUp.text = "Create Account"
                    Toast.makeText(
                        this,
                        "Registration failed: ${task.exception?.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
    }

    private fun saveUserDataToFirestore(userId: String, name: String, email: String) {
        val userData = hashMapOf(
            "userId" to userId,
            "name" to name,
            "email" to email,
            "createdAt" to System.currentTimeMillis(),
            "lastLogin" to System.currentTimeMillis(),
            "profileCompleted" to true
        )

        firestore.collection("users").document(userId)
            .set(userData)
            .addOnSuccessListener {
                binding.btnSignUp.isEnabled = true
                binding.btnSignUp.text = "Create Account"

                // Sign out user after signup so they go to login page
                auth.signOut()

                Toast.makeText(
                    this,
                    "Account created successfully! Please login.",
                    Toast.LENGTH_SHORT
                ).show()

                // Navigate to Login page
                navigateToLogin()
            }
            .addOnFailureListener { e ->
                binding.btnSignUp.isEnabled = true
                binding.btnSignUp.text = "Create Account"

                // Still sign out and navigate even if Firestore fails
                auth.signOut()

                Toast.makeText(
                    this,
                    "Account created! Please login.",
                    Toast.LENGTH_SHORT
                ).show()

                navigateToLogin()
            }
    }

    private fun validateInput(
        fullName: String,
        email: String,
        password: String,
        confirmPassword: String
    ): Boolean {
        if (fullName.isEmpty()) {
            Toast.makeText(this, "Please enter your full name", Toast.LENGTH_SHORT).show()
            binding.etFullName.requestFocus()
            return false
        }

        if (fullName.length < 3) {
            Toast.makeText(this, "Name must be at least 3 characters", Toast.LENGTH_SHORT).show()
            binding.etFullName.requestFocus()
            return false
        }

        if (email.isEmpty()) {
            Toast.makeText(this, "Please enter email", Toast.LENGTH_SHORT).show()
            binding.etEmail.requestFocus()
            return false
        }

        if (!Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            Toast.makeText(this, "Please enter a valid email", Toast.LENGTH_SHORT).show()
            binding.etEmail.requestFocus()
            return false
        }

        if (password.isEmpty()) {
            Toast.makeText(this, "Please enter password", Toast.LENGTH_SHORT).show()
            binding.etPassword.requestFocus()
            return false
        }

        if (password.length < 6) {
            Toast.makeText(
                this,
                "Password must be at least 6 characters",
                Toast.LENGTH_SHORT
            ).show()
            binding.etPassword.requestFocus()
            return false
        }

        if (confirmPassword.isEmpty()) {
            Toast.makeText(this, "Please confirm password", Toast.LENGTH_SHORT).show()
            binding.etConfirmPassword.requestFocus()
            return false
        }

        if (password != confirmPassword) {
            Toast.makeText(this, "Passwords do not match", Toast.LENGTH_SHORT).show()
            binding.etConfirmPassword.requestFocus()
            return false
        }

        return true
    }

    private fun signUpWithGoogle() {
        val signInIntent = googleSignInClient.signInIntent
        startActivityForResult(signInIntent, RC_SIGN_IN)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == RC_SIGN_IN) {
            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
            handleSignInResult(task)
        }
    }

    private fun handleSignInResult(completedTask: Task<GoogleSignInAccount>) {
        try {
            val account = completedTask.getResult(ApiException::class.java)
            firebaseAuthWithGoogle(account.idToken!!)
        } catch (e: ApiException) {
            Toast.makeText(
                this,
                "Google sign in failed: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun firebaseAuthWithGoogle(idToken: String) {
        val credential = GoogleAuthProvider.getCredential(idToken, null)
        auth.signInWithCredential(credential)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser
                    user?.let {
                        val name = it.displayName ?: "User"
                        val email = it.email ?: ""

                        // Check if this is a new user or existing user
                        checkIfNewGoogleUser(it.uid, name, email)
                    }
                } else {
                    Toast.makeText(
                        this,
                        "Authentication failed: ${task.exception?.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
    }

    private fun checkIfNewGoogleUser(userId: String, name: String, email: String) {
        firestore.collection("users").document(userId)
            .get()
            .addOnSuccessListener { document ->
                if (!document.exists()) {

                    saveUserDataToFirestore(userId, name, email)
                } else {

                    auth.signOut()
                    Toast.makeText(
                        this,
                        "Account already exists! Please login.",
                        Toast.LENGTH_SHORT
                    ).show()
                    navigateToLogin()
                }
            }
            .addOnFailureListener {

                auth.signOut()
                navigateToLogin()
            }
    }

    private fun navigateToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }
}