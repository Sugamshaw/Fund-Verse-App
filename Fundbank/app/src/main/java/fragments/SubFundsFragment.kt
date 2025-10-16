package fragments

import adapters.SubFundAdapter
import android.app.Dialog
import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import api.RetrofitClient
import com.example.fundbank.MainActivity
import com.example.fundbank.R
import com.example.fundbank.RefreshableFragment
import com.example.fundbank.databinding.FragmentDialogSubFundsBinding
import com.example.fundbank.databinding.FragmentSubFundsBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.launch
import models.SubFund

class SubFundsFragment : Fragment(), RefreshableFragment {

    private var _binding: FragmentSubFundsBinding? = null
    private val binding get() = _binding!!

    private lateinit var adapter: SubFundAdapter
    private var allSubFunds: List<SubFund> = emptyList()

    private var searchQuery: String? = null
    private var filterField: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSubFundsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        setupSearch()
        setupSwipeRefresh()
        setupFab()

        searchQuery = arguments?.getString("search_query")
        filterField = arguments?.getString("filter_field")

        loadData()
    }

    private fun setupRecyclerView() {
        adapter = SubFundAdapter(
            onEditClick = { showEditDialog(it) },
            onParentFundClick = { navigateToParentFund(it) },
            onMgmtIdClick = { navigateToManagement(it) },
            onLeIdClick = { navigateToLegalEntities(it) },
            onShareClassClick = { navigateToShareclassFunds(it) }
        )

        binding.recyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@SubFundsFragment.adapter
        }
    }

    private fun setupSearch() {
        binding.searchEditText.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                filterData(s.toString())
            }
            override fun afterTextChanged(s: Editable?) {}
        })
    }

    private fun setupSwipeRefresh() {
        binding.swipeRefresh.setColorSchemeResources(R.color.blue_600)
        binding.swipeRefresh.setOnRefreshListener {
            searchQuery = null
            filterField = null
            binding.filterInfoCard.visibility = View.GONE
            loadData()
        }
    }

    private fun setupFab() {
        binding.fabAdd.setOnClickListener { showAddDialog() }

        // Shrink/Extend FAB on scroll
        binding.recyclerView.addOnScrollListener(object : androidx.recyclerview.widget.RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: androidx.recyclerview.widget.RecyclerView, dx: Int, dy: Int) {
                if (dy > 0) {
                    binding.fabAdd.shrink()
                } else if (dy < 0) {
                    binding.fabAdd.extend()
                }
            }
        })
    }

    private fun filterData(query: String) {
        val filtered = if (query.isEmpty()) allSubFunds else allSubFunds.filter {
            it.subfundId.contains(query, ignoreCase = true) ||
                    it.parentFundId.contains(query, ignoreCase = true) ||
                    it.leId.contains(query, ignoreCase = true) ||
                    it.mgmtId.contains(query, ignoreCase = true) ||
                    it.isinSub.contains(query, ignoreCase = true) ||
                    it.currency.contains(query, ignoreCase = true)
        }
        adapter.submitList(filtered)
        updateEmptyState(filtered.isEmpty())
    }

    private fun loadData() {
        if (!isAdded || _binding == null) return

        binding.swipeRefresh.isRefreshing = true

        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.getSubFunds()

                if (!isAdded || _binding == null) return@launch

                if (response.isSuccessful && response.body() != null) {
                    allSubFunds = response.body() ?: emptyList()

                    val filteredList = when {
                        !searchQuery.isNullOrBlank() && filterField == "MGMT_ID" ->
                            allSubFunds.filter { it.mgmtId.equals(searchQuery, ignoreCase = true) }

                        !searchQuery.isNullOrBlank() && filterField == "LE_ID" ->
                            allSubFunds.filter { it.leId.equals(searchQuery, ignoreCase = true) }

                        !searchQuery.isNullOrBlank() && filterField == "PARENT_FUND_ID" ->
                            allSubFunds.filter { it.parentFundId.equals(searchQuery, ignoreCase = true) }

                        else -> allSubFunds
                    }

                    adapter.submitList(filteredList)
                    updateEmptyState(filteredList.isEmpty())

                    if (!searchQuery.isNullOrBlank()) {
                        binding.filterInfoCard.visibility = View.VISIBLE
                        val field = filterField ?: "UNKNOWN"
                        binding.filterInfo.text = "Filtered by: $field = $searchQuery"

                        binding.btnClearFilter.setOnClickListener {
                            searchQuery = null
                            filterField = null
                            adapter.submitList(allSubFunds)
                            binding.filterInfoCard.visibility = View.GONE
                        }
                    } else {
                        binding.filterInfoCard.visibility = View.GONE
                    }

                } else {
                    showError("Failed to load data: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            } finally {
                if (isAdded && _binding != null) {
                    binding.swipeRefresh.isRefreshing = false
                }
            }
        }
    }

    private fun showAddDialog() {
        val dialogBinding = FragmentDialogSubFundsBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        dialogBinding.tvDialogTitle.text = "Add SubFund"

        val nextId = generateNextSubFundId(allSubFunds)
        dialogBinding.etSubfundId.setText(nextId)
        dialogBinding.etSubfundId.isEnabled = false

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }

        dialogBinding.btnSubmit.text = "Submit"
        dialogBinding.btnSubmit.setOnClickListener {
            val subFund = SubFund(
                subfundId = nextId,
                parentFundId = dialogBinding.etParentFundId.text.toString().trim(),
                leId = dialogBinding.etLeId.text.toString().trim(),
                mgmtId = dialogBinding.etMgmtId.text.toString().trim(),
                isinSub = dialogBinding.etIsinSub.text.toString().trim(),
                currency = dialogBinding.etCurrency.text.toString().trim()
            )
            addSubFund(subFund)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun showEditDialog(entity: SubFund) {
        val dialogBinding = FragmentDialogSubFundsBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        dialogBinding.tvDialogTitle.text = "Edit SubFund"

        dialogBinding.etSubfundId.setText(entity.subfundId)
        dialogBinding.etSubfundId.isEnabled = false
        dialogBinding.etParentFundId.setText(entity.parentFundId)
        dialogBinding.etLeId.setText(entity.leId)
        dialogBinding.etMgmtId.setText(entity.mgmtId)
        dialogBinding.etIsinSub.setText(entity.isinSub)
        dialogBinding.etCurrency.setText(entity.currency)

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }
        dialogBinding.btnSubmit.text = "Update"

        dialogBinding.btnSubmit.setOnClickListener {
            val updated = entity.copy(
                parentFundId = dialogBinding.etParentFundId.text.toString().trim(),
                leId = dialogBinding.etLeId.text.toString().trim(),
                mgmtId = dialogBinding.etMgmtId.text.toString().trim(),
                isinSub = dialogBinding.etIsinSub.text.toString().trim(),
                currency = dialogBinding.etCurrency.text.toString().trim()
            )
            updateSubFund(updated)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun generateNextSubFundId(subFunds: List<SubFund>): String {
        if (subFunds.isEmpty()) return "SF000001"
        val max = subFunds.mapNotNull { it.subfundId.removePrefix("SF").toIntOrNull() }.maxOrNull() ?: 0
        return "SF" + String.format("%06d", max + 1)
    }

    private fun addSubFund(subFund: SubFund) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.addSubFund(subFund)
                if (response.isSuccessful) {
                    showSuccess("SubFund added successfully")
                    loadData()
                } else {
                    showError("Failed to add: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun updateSubFund(subFund: SubFund) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.updateSubFund(subFund.subfundId, subFund)
                if (response.isSuccessful) {
                    showSuccess("SubFund updated successfully")
                    loadData()
                } else {
                    showError("Failed to update: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun navigateToParentFund(fundId: String) {
        val fragment = FundsFragment()
        val bundle = Bundle().apply {
            putString("search_query", fundId)
            putString("filter_field", "FUND_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("funds", bundle)
    }

    private fun navigateToManagement(mgmtId: String) {
        val fragment = ManagementEntitiesFragment()
        val bundle = Bundle().apply {
            putString("search_query", mgmtId)
            putString("filter_field", "MGMT_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("management", bundle)
    }

    private fun navigateToLegalEntities(leId: String) {
        val fragment = LegalEntitiesFragment()
        val bundle = Bundle().apply {
            putString("search_query", leId)
            putString("filter_field", "LE_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("legal", bundle)
    }

    private fun navigateToShareclassFunds(parentFundId: String) {
        val fragment = ShareClassesFragment()
        val bundle = Bundle().apply {
            putString("search_query", parentFundId)
            putString("filter_field", "FUND_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("shareclass", bundle)
    }

    private fun updateEmptyState(isEmpty: Boolean) {
        if (!isAdded || _binding == null) return
        binding.emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
        binding.recyclerView.visibility = if (isEmpty) View.GONE else View.VISIBLE
    }

    private fun showSuccess(message: String) {
        if (!isAdded || _binding == null) return
        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT)
            .setBackgroundTint(Color.parseColor("#4CAF50"))
            .setTextColor(Color.WHITE)
            .show()
    }

    private fun showError(message: String) {
        if (!isAdded || _binding == null) return
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Retry") { loadData() }
            .show()
    }

    override fun refresh() {
        loadData()
    }

    override fun onDestroyView() {
        viewLifecycleOwner.lifecycleScope.coroutineContext.cancelChildren()
        _binding = null
        super.onDestroyView()
    }
}