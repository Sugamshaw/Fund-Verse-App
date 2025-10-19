package adapters


import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.adapter.FragmentStateAdapter
import fragments.OnboardingFragment

class OnboardingAdapter(fragmentActivity: FragmentActivity) : FragmentStateAdapter(fragmentActivity) {

    override fun getItemCount(): Int = 3

    override fun createFragment(position: Int): Fragment {
        return OnboardingFragment.newInstance(position)
    }
}
//class OnboardingAdapter(activity: FragmentActivity) : FragmentStateAdapter(activity) {
//
//    private val onboardingPages = listOf(
//        OnboardingPage(
//            "Welcome to FundVerse",
//            "Your comprehensive fund management solution for tracking and managing investment portfolios",
//            R.drawable.ic_onboarding_welcome
//        ),
//        OnboardingPage(
//            "Manage Multiple Entities",
//            "Easily organize legal entities, management companies, funds, and sub-funds in one centralized platform",
//            R.drawable.ic_onboarding_manage
//        ),
//        OnboardingPage(
//            "Real-Time Analytics",
//            "Track performance with interactive charts and detailed insights for informed decision-making",
//            R.drawable.ic_onboarding_analytics
//        ),
//        OnboardingPage(
//            "Secure & Reliable",
//            "Your data is protected with enterprise-grade security and cloud synchronization",
//            R.drawable.ic_onboarding_security
//        )
//    )
//
//    override fun getItemCount(): Int = onboardingPages.size
//
//    override fun createFragment(position: Int): Fragment {
//        return OnboardingFragment.newInstance(onboardingPages[position])
//    }
//}
//
//data class OnboardingPage(
//    val title: String,
//    val description: String,
//    val imageRes: Int
//)