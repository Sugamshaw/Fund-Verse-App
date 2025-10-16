package fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.fundbank.R
import com.example.fundbank.databinding.FragmentOnboardingBinding

class OnboardingFragment : Fragment() {

    private var _binding: FragmentOnboardingBinding? = null
    private val binding get() = _binding!!

    private var position: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            position = it.getInt(ARG_POSITION)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentOnboardingBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        when (position) {
            0 -> {
                binding.ivOnboarding.setImageResource(R.drawable.managefund)
                binding.tvTitle.text = getString(R.string.onboarding_title_1)
                binding.tvDescription.text = getString(R.string.onboarding_desc_1)
            }
            1 -> {
                binding.ivOnboarding.setImageResource(R.drawable.trackperformance)
                binding.tvTitle.text = getString(R.string.onboarding_title_2)
                binding.tvDescription.text = getString(R.string.onboarding_desc_2)
            }
            2 -> {
                binding.ivOnboarding.setImageResource(R.drawable.secureandreliable)
                binding.tvTitle.text = getString(R.string.onboarding_title_3)
                binding.tvDescription.text = getString(R.string.onboarding_desc_3)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null // Prevent memory leaks
    }

    companion object {
        private const val ARG_POSITION = "position"

        fun newInstance(position: Int) = OnboardingFragment().apply {
            arguments = Bundle().apply {
                putInt(ARG_POSITION, position)
            }
        }
    }
}
