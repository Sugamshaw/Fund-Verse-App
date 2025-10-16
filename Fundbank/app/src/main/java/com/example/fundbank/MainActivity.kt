package com.example.fundbank

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.fragment.app.Fragment
import charts.ChartActivity
import com.example.fundbank.databinding.ActivityMainBinding
import fragments.FundsFragment
import fragments.LegalEntitiesFragment
import fragments.ManagementEntitiesFragment
import fragments.ShareClassesFragment
import fragments.SubFundsFragment



class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var currentFragment: String = "legal"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.title = "Fund Management System"

        setupBottomNavigation()

        // Load initial fragment
        if (savedInstanceState == null) {
            loadFragment(LegalEntitiesFragment(), "legal")
        }
    }

    private fun setupBottomNavigation() {
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_legal -> {
                    loadFragment(LegalEntitiesFragment(), "legal")
                    updateTitle("Legal Entity")
                    true
                }
                R.id.nav_management -> {
                    loadFragment(ManagementEntitiesFragment(), "management")
                    updateTitle("Management Entity")
                    true
                }
                R.id.nav_funds -> {
                    loadFragment(FundsFragment(), "funds")
                    updateTitle("Fund Master")
                    true
                }
                R.id.nav_subfunds -> {
                    loadFragment(SubFundsFragment(), "subfunds")
                    updateTitle("Sub Funds")
                    true
                }
                R.id.nav_shareclass -> {
                    loadFragment(ShareClassesFragment(), "shareclass")
                    updateTitle("Share Class")
                    true
                }
                else -> false
            }
        }
    }
    private fun updateTitle(title: String) {
        binding.toolbarTitle.animate()
            .alpha(0f)
            .setDuration(100)
            .withEndAction {
                binding.toolbarTitle.text = title
                binding.toolbarTitle.animate()
                    .alpha(1f)
                    .setDuration(100)
                    .start()
            }
            .start()
    }
    private fun loadFragment(fragment: Fragment, tag: String) {
        currentFragment = tag
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment, tag)
            .commit()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_charts -> {
                startActivity(Intent(this, ChartActivity::class.java))
                true
            }
            R.id.action_refresh -> {
                val fragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
                if (fragment is RefreshableFragment) {
                    fragment.refresh()
                }
                true
            }
            R.id.action_setting -> {
                startActivity(Intent(this, SettingsActivity::class.java))
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
    // In MainActivity.kt, update the logout function in SettingsActivity navigation:

//    private fun logout() {
//        // Sign out from Firebase
//        auth.signOut()
//
//        // Clear login state
//        val sharedPref = getSharedPreferences("FundBankPrefs", MODE_PRIVATE)
//        sharedPref.edit().apply {
//            putBoolean("isLoggedIn", false)
//            apply()
//        }
//
//        // Navigate to Login
//        val intent = Intent(this, LoginActivity::class.java)
//        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
//        startActivity(intent)
//        finish()
//    }
//    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
//        menuInflater.inflate(R.menu.main_menu, menu)
//        return true
//    }
//
//    override fun onOptionsItemSelected(item: MenuItem): Boolean {
//        return when (item.itemId) {
//            R.id.action_charts -> {
//                startActivity(Intent(this, ChartActivity::class.java))
//                true
//            }
//            R.id.action_refresh -> {
//                val fragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
//                if (fragment is RefreshableFragment) {
//                    fragment.refresh()
//                }
//                true
//            }
//            else -> super.onOptionsItemSelected(item)
//        }
//    }
    fun openFragmentFromChild(tag: String, bundle: Bundle? = null) {
        val fragment: Fragment = when (tag) {
            "management" -> ManagementEntitiesFragment()
            "funds" -> FundsFragment()
            "subfunds" -> SubFundsFragment()
            "shareclass" -> ShareClassesFragment()
            else -> LegalEntitiesFragment()
        }
        if (bundle != null) fragment.arguments = bundle

        // Update bottom navigation selection
        val navId = when (tag) {
            "legal" -> R.id.nav_legal
            "management" -> R.id.nav_management
            "funds" -> R.id.nav_funds
            "subfunds" -> R.id.nav_subfunds
            "shareclass" -> R.id.nav_shareclass
            else -> R.id.nav_legal
        }

        binding.bottomNavigation.selectedItemId = navId

        // Replace fragment
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment, tag)
            .addToBackStack(null)
            .commit()
    }

}

interface RefreshableFragment {
    fun refresh()
}
