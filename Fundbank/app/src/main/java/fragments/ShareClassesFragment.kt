package fragments

import adapters.ShareClassAdapter
import android.app.Dialog
import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import api.RetrofitClient
import com.example.fundbank.MainActivity
import com.example.fundbank.R
import com.example.fundbank.RefreshableFragment
import com.example.fundbank.databinding.FragmentDialogShareClassesBinding
import com.example.fundbank.databinding.FragmentShareClassesBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.launch
import models.ShareClass

class ShareClassesFragment : Fragment(), RefreshableFragment {

    private var _binding: FragmentShareClassesBinding? = null
    private val binding get() = _binding!!

    private lateinit var adapter: ShareClassAdapter
    private var allShareClasses: List<ShareClass> = emptyList()
    private var searchQuery: String? = null
    private var filterField: String? = null
    private var currentStatusFilter = "ALL"

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentShareClassesBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        setupStatusFilter()
        setupSearch()
        setupFab()
        setupSwipeRefresh()
        setupSortButtons()

        searchQuery = arguments?.getString("search_query")
        filterField = arguments?.getString("filter_field")

        loadData()
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

    private fun setupSortButtons() {
        binding.btnSortAum.setOnClickListener { sortByAUM() }
        binding.btnSortNav.setOnClickListener { sortByNAV() }
        binding.btnSortFee.setOnClickListener { sortByFee() }
    }

    private fun sortByAUM() {
        adapter.submitList(adapter.currentList.sortedByDescending { it.aum })
        showSuccess("Sorted by AUM")
    }

    private fun sortByNAV() {
        adapter.submitList(adapter.currentList.sortedByDescending { it.nav })
        showSuccess("Sorted by NAV")
    }

    private fun sortByFee() {
        adapter.submitList(adapter.currentList.sortedByDescending { it.feeMgmt })
        showSuccess("Sorted by Fee")
    }

    private fun setupStatusFilter() {
        val statuses = arrayOf("ALL", "ACTIVE", "CLOSED")
        val spinnerAdapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, statuses)
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerStatus.adapter = spinnerAdapter

        binding.spinnerStatus.setOnItemSelectedListener(object :
            android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                currentStatusFilter = statuses[position]
                val query = binding.searchEditText.text.toString()
                filterData(query)
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        })
    }

    private fun setupRecyclerView() {
        adapter = ShareClassAdapter(
            onEditClick = { showEditDialog(it) },
            onFundIdClick = { navigateToFund(it) }
        )
        binding.recyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@ShareClassesFragment.adapter
        }
    }

    private fun showAddDialog() {
        val dialogBinding = FragmentDialogShareClassesBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        dialogBinding.tvDialogTitle.text = "Add Share Class"

        val nextId = generateNextScId(allShareClasses)
        dialogBinding.etScId.setText(nextId)
        dialogBinding.etScId.isEnabled = false

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }

        dialogBinding.btnSubmit.text = "Submit"
        dialogBinding.btnSubmit.setOnClickListener {
            val shareClass = ShareClass(
                scId = nextId,
                fundId = dialogBinding.etFundId.text.toString().trim(),
                isinSc = dialogBinding.etIsinSc.text.toString().trim(),
                currency = dialogBinding.etCurrency.text.toString().trim(),
                distribution = dialogBinding.etDistribution.text.toString().trim(),
                feeMgmt = dialogBinding.etFeeMgmt.text.toString().toDoubleOrNull() ?: 0.0,
                perfFee = dialogBinding.etPerfFee.text.toString().toDoubleOrNull() ?: 0.0,
                expenseRatio = dialogBinding.etExpenseRatio.text.toString().toDoubleOrNull() ?: 0.0,
                nav = dialogBinding.etNav.text.toString().toDoubleOrNull() ?: 0.0,
                aum = dialogBinding.etAum.text.toString().toDoubleOrNull() ?: 0.0,
                status = dialogBinding.etStatus.text.toString().trim()
            )

            addShareClass(shareClass)
            dialog.dismiss()
        }
        dialog.show()
    }

    private fun showEditDialog(shareClass: ShareClass) {
        val dialogBinding = FragmentDialogShareClassesBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        dialogBinding.tvDialogTitle.text = "Edit Share Class"
        dialogBinding.etScId.setText(shareClass.scId)
        dialogBinding.etScId.isEnabled = false
        dialogBinding.etFundId.setText(shareClass.fundId)
        dialogBinding.etIsinSc.setText(shareClass.isinSc)
        dialogBinding.etCurrency.setText(shareClass.currency)
        dialogBinding.etDistribution.setText(shareClass.distribution)
        dialogBinding.etFeeMgmt.setText(shareClass.feeMgmt.toString())
        dialogBinding.etPerfFee.setText(shareClass.perfFee.toString())
        dialogBinding.etExpenseRatio.setText(shareClass.expenseRatio.toString())
        dialogBinding.etNav.setText(shareClass.nav.toString())
        dialogBinding.etAum.setText(shareClass.aum.toString())
        dialogBinding.etStatus.setText(shareClass.status)

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }

        dialogBinding.btnSubmit.text = "Update"
        dialogBinding.btnSubmit.setOnClickListener {
            val updated = shareClass.copy(
                fundId = dialogBinding.etFundId.text.toString().trim(),
                isinSc = dialogBinding.etIsinSc.text.toString().trim(),
                currency = dialogBinding.etCurrency.text.toString().trim(),
                distribution = dialogBinding.etDistribution.text.toString().trim(),
                feeMgmt = dialogBinding.etFeeMgmt.text.toString().toDoubleOrNull() ?: 0.0,
                perfFee = dialogBinding.etPerfFee.text.toString().toDoubleOrNull() ?: 0.0,
                expenseRatio = dialogBinding.etExpenseRatio.text.toString().toDoubleOrNull() ?: 0.0,
                nav = dialogBinding.etNav.text.toString().toDoubleOrNull() ?: 0.0,
                aum = dialogBinding.etAum.text.toString().toDoubleOrNull() ?: 0.0,
                status = dialogBinding.etStatus.text.toString().trim()
            )

            updateShareClass(updated)
            dialog.dismiss()
        }
        dialog.show()
    }

    private fun generateNextScId(shareClasses: List<ShareClass>): String {
        if (shareClasses.isEmpty()) return "SC000001"
        val max = shareClasses.mapNotNull { it.scId.removePrefix("SC").toIntOrNull() }.maxOrNull() ?: 0
        return "SC" + String.format("%06d", max + 1)
    }

    private fun addShareClass(shareClass: ShareClass) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.addShareClass(shareClass)
                if (response.isSuccessful) {
                    showSuccess("Share Class added successfully")
                    loadData()
                } else showError("Failed to add: ${response.message()}")
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun updateShareClass(shareClass: ShareClass) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.updateShareClass(shareClass.scId, shareClass)
                if (response.isSuccessful) {
                    showSuccess("Share Class updated successfully")
                    loadData()
                } else showError("Failed to update: ${response.message()}")
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun navigateToFund(fundId: String) {
        val fragment = FundsFragment()
        val bundle = Bundle().apply {
            putString("search_query", fundId)
            putString("filter_field", "FUND_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("funds", bundle)
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

    private fun filterData(query: String) {
        var filtered = allShareClasses
        if (currentStatusFilter != "ALL") {
            filtered = filtered.filter { it.status.equals(currentStatusFilter, ignoreCase = true) }
        }

        if (query.isNotEmpty()) {
            filtered = filtered.filter {
                it.scId.contains(query, true) || it.fundId.contains(query, true) || it.currency.contains(query, true)
            }
        }

        adapter.submitList(filtered)
        updateEmptyState(filtered.isEmpty())
    }

    private fun loadData() {
        if (!isAdded || _binding == null) return
        binding.swipeRefresh.isRefreshing = true

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.getShareClasses()
                if (!isAdded || _binding == null) return@launch

                if (response.isSuccessful && response.body() != null) {
                    allShareClasses = response.body() ?: emptyList()

                    val filteredList = when {
                        !searchQuery.isNullOrBlank() && filterField == "FUND_ID" ->
                            allShareClasses.filter { it.fundId.equals(searchQuery, ignoreCase = true) }

                        !searchQuery.isNullOrBlank() && filterField == "SC_ID" ->
                            allShareClasses.filter { it.scId.equals(searchQuery, ignoreCase = true) }

                        !searchQuery.isNullOrBlank() && filterField == "STATUS" ->
                            allShareClasses.filter { it.status.equals(searchQuery, ignoreCase = true) }

                        else -> allShareClasses
                    }

                    if (!isAdded || _binding == null) return@launch
                    adapter.submitList(filteredList)
                    updateEmptyState(filteredList.isEmpty())

                    if (!searchQuery.isNullOrBlank()) showFilterBanner()
                    else binding.filterInfoCard.visibility = View.GONE
                } else {
                    if (isAdded && _binding != null)
                        showError("Failed to load data: ${response.message()}")
                }
            } catch (e: Exception) {
                if (isAdded && _binding != null)
                    showError("Error: ${e.message}")
            } finally {
                if (isAdded && _binding != null) {
                    binding.swipeRefresh.isRefreshing = false
                }
            }
        }
    }

    private fun showFilterBanner() {
        binding.filterInfoCard.visibility = View.VISIBLE
        val field = filterField ?: "UNKNOWN"
        binding.filterInfo.text = "Filtered by: $field = $searchQuery"

        binding.btnClearFilter.setOnClickListener {
            searchQuery = null
            filterField = null
            adapter.submitList(allShareClasses)
            binding.filterInfoCard.visibility = View.GONE
        }
    }

    private fun updateEmptyState(isEmpty: Boolean) {
        binding.emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
        binding.recyclerView.visibility = if (isEmpty) View.GONE else View.VISIBLE
    }

    private fun showSuccess(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT)
            .setBackgroundTint(Color.parseColor("#4CAF50"))
            .setTextColor(Color.WHITE)
            .show()
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Retry") { loadData() }
            .show()
    }

    override fun refresh() = loadData()

    override fun onDestroyView() {
        viewLifecycleOwner.lifecycleScope.coroutineContext.cancelChildren()
        _binding = null
        super.onDestroyView()
    }
}

//package fragments
//
//import adapters.ShareClassAdapter
//import android.graphics.Color
//import android.os.Bundle
//import android.text.SpannableString
//import android.text.style.ForegroundColorSpan
//import android.text.style.UnderlineSpan
//import androidx.fragment.app.Fragment
//import android.view.LayoutInflater
//import android.view.View
//import android.view.ViewGroup
//import android.widget.ArrayAdapter
//import android.widget.SearchView
//import android.widget.TextView
//import androidx.lifecycle.lifecycleScope
//import androidx.recyclerview.widget.LinearLayoutManager
//import api.RetrofitClient
//import com.example.fundbank.R
//import com.example.fundbank.RefreshableFragment
//import com.example.fundbank.databinding.FragmentShareClassesBinding
//import com.google.android.material.snackbar.Snackbar
//import kotlinx.coroutines.launch
//import models.ShareClass
//
//class ShareClassesFragment : Fragment(), RefreshableFragment {
//    private var _binding: FragmentShareClassesBinding? = null
//    private val binding get() = _binding!!
//
//    private lateinit var adapter: ShareClassAdapter
//    private var allShareClasses: List<ShareClass> = emptyList()
//    private var currentStatusFilter = "ALL"
//
//    private var searchQuery: String? = null
//    private var filterField: String? = null
//
//    override fun onCreateView(
//        inflater: LayoutInflater,
//        container: ViewGroup?,
//        savedInstanceState: Bundle?
//    ): View {
//        _binding = FragmentShareClassesBinding.inflate(inflater, container, false)
//        return binding.root
//    }
//
//    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
//        super.onViewCreated(view, savedInstanceState)
//
//        setupRecyclerView()
//        setupSearch()
//        setupStatusFilter()
//        setupSwipeRefresh()
//        setupSortButtons()
//        searchQuery = arguments?.getString("search_query")
//        filterField = arguments?.getString("filter_field")
//
//        loadData()
//    }
//
//    private fun setupRecyclerView() {
//        adapter = ShareClassAdapter()
//        binding.recyclerView.apply {
//            layoutManager = LinearLayoutManager(context)
//            adapter = this@ShareClassesFragment.adapter
//        }
//    }
//
//    private fun setupSearch() {
//        binding.searchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
//            override fun onQueryTextSubmit(query: String?) = false
//
//            override fun onQueryTextChange(newText: String?): Boolean {
//                filterData()
//                return true
//            }
//        })
//    }
//
//    private fun setupStatusFilter() {
//        val statuses = arrayOf("ALL", "ACTIVE", "CLOSED")
//        val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, statuses)
//        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
//
//        binding.spinnerStatus.adapter = adapter
//        binding.spinnerStatus.setOnItemSelectedListener(object : android.widget.AdapterView.OnItemSelectedListener {
//            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
//                currentStatusFilter = statuses[position]
//                filterData()
//            }
//
//            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
//        })
//    }
//
//    private fun setupSortButtons() {
//        binding.btnSortAum.setOnClickListener {
//            sortByAUM()
//        }
//
//        binding.btnSortNav.setOnClickListener {
//            sortByNAV()
//        }
//
//        binding.btnSortFee.setOnClickListener {
//            sortByFee()
//        }
//    }
//
//    private fun setupSwipeRefresh() {
//        binding.swipeRefresh.setOnRefreshListener {
//            searchQuery = null
//            filterField = null
//            loadData()
//        }
//    }
//
//    private fun sortByAUM() {
//        val sorted = adapter.currentList.sortedByDescending { it.aum }
//        adapter.submitList(sorted)
//    }
//
//    private fun sortByNAV() {
//        val sorted = adapter.currentList.sortedByDescending { it.nav }
//        adapter.submitList(sorted)
//    }
//
//    private fun sortByFee() {
//        val sorted = adapter.currentList.sortedByDescending { it.feeMgmt }
//        adapter.submitList(sorted)
//    }
//    private fun loadData() {
//        // Avoid running if fragment is detached or view not ready
//        if (!isAdded || _binding == null) return
//
//        binding.swipeRefresh.isRefreshing = true
//        binding.progressBar.visibility = View.VISIBLE
//
//        lifecycleScope.launch {
//            try {
//                val response = RetrofitClient.apiService.getShareClasses()
//
//                // Recheck if fragment is still active after network response
//                if (!isAdded || _binding == null) return@launch
//
//                if (response.isSuccessful && response.body() != null) {
//                    allShareClasses = response.body() ?: emptyList()
//
//                    // Apply filter if search query or field is provided
//                    val filteredList = when {
//                        !searchQuery.isNullOrBlank() && filterField == "FUND_ID" ->
//                            allShareClasses.filter { it.fundId.equals(searchQuery, ignoreCase = true) }
//
//                        !searchQuery.isNullOrBlank() && filterField == "SC_ID" ->
//                            allShareClasses.filter { it.scId.equals(searchQuery, ignoreCase = true) }
//
//                        !searchQuery.isNullOrBlank() && filterField == "STATUS" ->
//                            allShareClasses.filter { it.status.equals(searchQuery, ignoreCase = true) }
//
//                        else -> allShareClasses
//                    }
//
//                    adapter.submitList(filteredList)
//                    updateEmptyState(filteredList.isEmpty())
//
//                    // Show active filter banner if applicable
//                    if (!searchQuery.isNullOrBlank()) {
//                        val field = filterField ?: "UNKNOWN"
//                        val text = "Filtered by: $field = $searchQuery   Clear All Filters"
//                        val spannable = SpannableString(text)
//                        val start = text.indexOf("Clear All Filters")
//                        val end = start + "Clear All Filters".length
//                        spannable.setSpan(UnderlineSpan(), start, end, 0)
//                        spannable.setSpan(ForegroundColorSpan(Color.BLUE), start, end, 0)
//
//                        binding.filterInfo.apply {
//                            visibility = View.VISIBLE
//                            setText(spannable, TextView.BufferType.SPANNABLE)
//                            setOnClickListener {
//                                // Clear filters when clicked
//                                searchQuery = null
//                                filterField = null
//                                adapter.submitList(allShareClasses)
//                                visibility = View.GONE
//                            }
//                        }
//                    } else {
//                        binding.filterInfo.visibility = View.GONE
//                    }
//                } else {
//                    if (isAdded && _binding != null) {
//                        showError("Failed to load data: ${response.message()}")
//                    }
//                }
//            } catch (e: Exception) {
//                if (isAdded && _binding != null) {
//                    showError("Error: ${e.message}")
//                }
//            } finally {
//                if (isAdded && _binding != null) {
//                    binding.swipeRefresh.isRefreshing = false
//                    binding.progressBar.visibility = View.GONE
//                }
//            }
//        }
//    }
//
//    private fun filterData() {
//        val query = binding.searchView.query.toString()
//
//        var filtered = allShareClasses
//
//        // Apply status filter
//        if (currentStatusFilter != "ALL") {
//            filtered = filtered.filter { it.status == currentStatusFilter }
//        }
//
//        // Apply search filter
//        if (query.isNotEmpty()) {
//            filtered = filtered.filter {
//                it.scId.contains(query, ignoreCase = true) ||
//                        it.fundId.contains(query, ignoreCase = true) ||
//                        it.currency.contains(query, ignoreCase = true)
//            }
//        }
//
//        adapter.submitList(filtered)
//        updateEmptyState(filtered.isEmpty())
//    }
////
////    private fun loadData() {
////        binding.swipeRefresh.isRefreshing = true
////        binding.progressBar.visibility = View.VISIBLE
////
////        lifecycleScope.launch {
////            try {
////                val response = RetrofitClient.apiService.getShareClasses()
////                if (response.isSuccessful && response.body() != null) {
////                    allShareClasses = response.body()!!
////                    filterData()
////                } else {
////                    showError("Failed to load data: ${response.message()}")
////                }
////            } catch (e: Exception) {
////                showError("Error: ${e.message}")
////            } finally {
////                binding.swipeRefresh.isRefreshing = false
////                binding.progressBar.visibility = View.GONE
////            }
////        }
////    }
//
//    private fun updateEmptyState(isEmpty: Boolean) {
//        binding.emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
//        binding.recyclerView.visibility = if (isEmpty) View.GONE else View.VISIBLE
//    }
//
//    private fun showError(message: String) {
//        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
//            .setAction("Retry") { loadData() }
//            .show()
//    }
//
//    override fun refresh() {
//        loadData()
//    }
//
//    override fun onDestroyView() {
//        super.onDestroyView()
//        _binding = null
//    }
//}