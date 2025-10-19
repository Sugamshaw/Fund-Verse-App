package fragments

import adapters.FundAdapter
import android.app.Dialog
import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
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
import com.example.fundbank.databinding.FragmentDialogFundBinding
import com.example.fundbank.databinding.FragmentFundsBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.launch
import models.FundMaster

class FundsFragment : Fragment(), RefreshableFragment {

    private var _binding: FragmentFundsBinding? = null
    private val binding get() = _binding!!

    private lateinit var adapter: FundAdapter
    private var allFunds: List<FundMaster> = emptyList()

    private var searchQuery: String? = null
    private var filterField: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentFundsBinding.inflate(inflater, container, false)
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
        adapter = FundAdapter(
            onEditClick = { entity -> showEditDialog(entity) },
            onSubFundsClick = { fundId -> navigateToSubFunds(fundId) },
            onShareclassClick = { fundId -> navigateToShareclassFunds(fundId) },
            onLeIdClick = { leId -> navigateToLegalEntities(leId) },
            onMgmtIdClick = { mgmtId -> navigateToManagement(mgmtId) }
        )
        binding.recyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@FundsFragment.adapter
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
        val filtered = if (query.isEmpty()) allFunds else allFunds.filter {
            it.fundName.contains(query, ignoreCase = true) ||
                    it.fundId.contains(query, ignoreCase = true) ||
                    it.mgmtId.contains(query, ignoreCase = true) ||
                    it.leId.contains(query, ignoreCase = true) ||
                    it.fundType.contains(query, ignoreCase = true)
        }
        adapter.submitList(filtered)
        updateEmptyState(filtered.isEmpty())
    }

    private fun loadData() {
        if (!isAdded || _binding == null) return

        binding.swipeRefresh.isRefreshing = true

        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.getFunds()

                if (!isAdded || _binding == null) return@launch

                if (response.isSuccessful && response.body() != null) {
                    allFunds = response.body() ?: emptyList()

                    val filteredList = when {
                        !searchQuery.isNullOrBlank() && filterField == "MGMT_ID" ->
                            allFunds.filter { it.mgmtId.equals(searchQuery, ignoreCase = true) }

                        !searchQuery.isNullOrBlank() && filterField == "LE_ID" ->
                            allFunds.filter { it.leId.equals(searchQuery, ignoreCase = true) }

                        !searchQuery.isNullOrBlank() && filterField == "FUND_ID" ->
                            allFunds.filter { it.fundId.equals(searchQuery, ignoreCase = true) }

                        else -> allFunds
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
                            adapter.submitList(allFunds)
                            binding.filterInfoCard.visibility = View.GONE
                        }
                    } else {
                        binding.filterInfoCard.visibility = View.GONE
                    }
                } else {
                    if (isAdded && _binding != null) {
                        showError("Failed to load data: ${response.message()}")
                    }
                }
            } catch (e: Exception) {
                if (isAdded && _binding != null) {
                    showError("Error: ${e.message}")
                }
            } finally {
                if (isAdded && _binding != null) {
                    binding.swipeRefresh.isRefreshing = false
                }
            }
        }
    }

    private fun showAddDialog() {
        val dialogBinding = FragmentDialogFundBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        dialogBinding.tvDialogTitle.text = "Add Fund"

        val nextFundId = generateNextFundId(allFunds)
        dialogBinding.etFundId.setText(nextFundId)
        dialogBinding.etFundId.isEnabled = false

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }

        dialogBinding.btnSubmit.text = "Submit"
        dialogBinding.btnSubmit.setOnClickListener {
            val fund = FundMaster(
                fundId = nextFundId,
                mgmtId = dialogBinding.etMgmtId.text.toString().trim(),
                leId = dialogBinding.etLeId.text.toString().trim(),
                fundCode = dialogBinding.etFundCode.text.toString().trim(),
                fundName = dialogBinding.etFundName.text.toString().trim(),
                fundType = dialogBinding.etFundType.text.toString().trim(),
                baseCurrency = dialogBinding.etBaseCurrency.text.toString().trim(),
                domicile = dialogBinding.etDomicile.text.toString().trim(),
                isinMaster = dialogBinding.etIsinMaster.text.toString().trim(),
                status = dialogBinding.etStatus.text.toString().trim()
            )

            addFund(fund)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun showEditDialog(entity: FundMaster) {
        val dialogBinding = FragmentDialogFundBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        dialogBinding.tvDialogTitle.text = "Edit Fund"

        dialogBinding.etFundId.setText(entity.fundId)
        dialogBinding.etFundId.isEnabled = false
        dialogBinding.etMgmtId.setText(entity.mgmtId)
        dialogBinding.etLeId.setText(entity.leId)
        dialogBinding.etFundCode.setText(entity.fundCode)
        dialogBinding.etFundName.setText(entity.fundName)
        dialogBinding.etFundType.setText(entity.fundType)
        dialogBinding.etBaseCurrency.setText(entity.baseCurrency)
        dialogBinding.etDomicile.setText(entity.domicile)
        dialogBinding.etIsinMaster.setText(entity.isinMaster)
        dialogBinding.etStatus.setText(entity.status)

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }

        dialogBinding.btnSubmit.text = "Update"
        dialogBinding.btnSubmit.setOnClickListener {
            val updatedFund = entity.copy(
                mgmtId = dialogBinding.etMgmtId.text.toString().trim(),
                leId = dialogBinding.etLeId.text.toString().trim(),
                fundCode = dialogBinding.etFundCode.text.toString().trim(),
                fundName = dialogBinding.etFundName.text.toString().trim(),
                fundType = dialogBinding.etFundType.text.toString().trim(),
                baseCurrency = dialogBinding.etBaseCurrency.text.toString().trim(),
                domicile = dialogBinding.etDomicile.text.toString().trim(),
                isinMaster = dialogBinding.etIsinMaster.text.toString().trim(),
                status = dialogBinding.etStatus.text.toString().trim()
            )

            updateFund(updatedFund)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun generateNextFundId(funds: List<FundMaster>): String {
        if (funds.isEmpty()) return "F000001"

        val maxId = funds.mapNotNull { fund ->
            fund.fundId.removePrefix("F").toIntOrNull()
        }.maxOrNull() ?: 0

        return "F" + String.format("%06d", maxId + 1)
    }

    private fun addFund(fund: FundMaster) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.addFund(fund)
                if (response.isSuccessful) {
                    showSuccess("Fund added successfully")
                    loadData()
                } else {
                    showError("Failed to add: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun updateFund(fund: FundMaster) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.updateFund(fund.fundId, fund)
                if (response.isSuccessful) {
                    showSuccess("Fund updated successfully")
                    loadData()
                } else {
                    Log.d("FundsFragment", "Updating fund: ${fund.fundId}")
                    Log.d("FundsFragment", "Payload: $fund")
                    Log.d("FundsFragment", "Failed to update: ${response.message()}")
                    showError("Failed to update: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun navigateToSubFunds(fundId: String) {
        val bundle = Bundle().apply {
            putString("search_query", fundId)
            putString("filter_field", "PARENT_FUND_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("subfunds", bundle)
    }

    private fun navigateToShareclassFunds(fundId: String) {
        val bundle = Bundle().apply {
            putString("search_query", fundId)
            putString("filter_field", "FUND_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("shareclass", bundle)
    }

    private fun navigateToManagement(mgmtId: String) {
        val bundle = Bundle().apply {
            putString("search_query", mgmtId)
            putString("filter_field", "MGMT_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("management", bundle)
    }

    private fun navigateToLegalEntities(leId: String) {
        val bundle = Bundle().apply {
            putString("search_query", leId)
            putString("filter_field", "LE_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("legal", bundle)
    }

    private fun updateEmptyState(isEmpty: Boolean) {
        if (!isAdded || _binding == null) return
        binding.emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
        binding.recyclerView.visibility = if (isEmpty) View.GONE else View.VISIBLE
    }

    private fun showError(message: String) {
        if (!isAdded || _binding == null) return
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Retry") { loadData() }
            .show()
    }

    private fun showSuccess(message: String) {
        if (!isAdded || _binding == null) return
        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT)
            .setBackgroundTint(Color.parseColor("#4CAF50"))
            .setTextColor(Color.WHITE)
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