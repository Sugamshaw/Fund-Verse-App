package fragments

import adapters.ManagementEntityAdapter
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
import com.example.fundbank.databinding.FragmentDialogManagementBinding
import com.example.fundbank.databinding.FragmentManagementEntitiesBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.launch
import models.ManagementEntity

class ManagementEntitiesFragment : Fragment(), RefreshableFragment {
    private var _binding: FragmentManagementEntitiesBinding? = null
    private val binding get() = _binding!!

    private lateinit var adapter: ManagementEntityAdapter
    private var allEntities: List<ManagementEntity> = emptyList()

    private var searchQuery: String? = null
    private var filterField: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentManagementEntitiesBinding.inflate(inflater, container, false)
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
        adapter = ManagementEntityAdapter(
            onEditClick = { entity -> showEditDialog(entity) },
            onFundsClick = { mgmtId -> navigateToFunds(mgmtId) },
            onSubFundsClick = { mgmtId -> navigateToSubFunds(mgmtId) },
            onLeIdClick = { leId -> navigateToLegalEntities(leId) }
        )
        binding.recyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@ManagementEntitiesFragment.adapter
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
        val filtered = if (query.isEmpty()) allEntities else allEntities.filter {
            it.mgmtId.contains(query, ignoreCase = true) ||
                    it.leId.contains(query, ignoreCase = true) ||
                    it.registrationNo.contains(query, ignoreCase = true) ||
                    it.domicile.contains(query, ignoreCase = true) ||
                    it.entityType.contains(query, ignoreCase = true)
        }
        adapter.submitList(filtered)
        updateEmptyState(filtered.isEmpty())
    }

    private fun loadData() {
        if (!isAdded || _binding == null) return
        binding.swipeRefresh.isRefreshing = true

        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.getManagementEntities()
                if (!isAdded || _binding == null) return@launch

                if (response.isSuccessful && response.body() != null) {
                    allEntities = response.body()!!

                    val filteredList = if (!searchQuery.isNullOrBlank() && filterField == "LE_ID") {
                        allEntities.filter { it.leId.equals(searchQuery, ignoreCase = true) }
                    } else allEntities

                    adapter.submitList(filteredList)
                    updateEmptyState(filteredList.isEmpty())

                    if (!searchQuery.isNullOrBlank()) {
                        binding.filterInfoCard.visibility = View.VISIBLE
                        val field = filterField ?: "UNKNOWN"
                        binding.filterInfo.text = "Filtered by: $field = $searchQuery"

                        binding.btnClearFilter.setOnClickListener {
                            searchQuery = null
                            filterField = null
                            adapter.submitList(allEntities)
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
        val dialogBinding = FragmentDialogManagementBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        dialogBinding.tvDialogTitle.text = "Add Management Entity"

        val nextId = generateNextMgmtId(allEntities)
        dialogBinding.etMgmtId.setText(nextId)
        dialogBinding.etMgmtId.isEnabled = false

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }

        dialogBinding.btnSubmit.setOnClickListener {
            val entity = ManagementEntity(
                mgmtId = nextId,
                leId = dialogBinding.etLeId.text.toString().trim(),
                registrationNo = dialogBinding.etRegistration.text.toString().trim(),
                domicile = dialogBinding.etDomicile.text.toString().trim(),
                entityType = dialogBinding.etEntityType.text.toString().trim()
            )
            addManagementEntity(entity)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun showEditDialog(entity: ManagementEntity) {
        val dialogBinding = FragmentDialogManagementBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        dialogBinding.tvDialogTitle.text = "Edit Management Entity"
        dialogBinding.etMgmtId.setText(entity.mgmtId)
        dialogBinding.etMgmtId.isEnabled = false
        dialogBinding.etLeId.setText(entity.leId)
        dialogBinding.etRegistration.setText(entity.registrationNo)
        dialogBinding.etDomicile.setText(entity.domicile)
        dialogBinding.etEntityType.setText(entity.entityType)

        dialogBinding.btnCancel.setOnClickListener { dialog.dismiss() }
        dialogBinding.btnSubmit.text = "Update"

        dialogBinding.btnSubmit.setOnClickListener {
            val updatedEntity = entity.copy(
                leId = dialogBinding.etLeId.text.toString().trim(),
                registrationNo = dialogBinding.etRegistration.text.toString().trim(),
                domicile = dialogBinding.etDomicile.text.toString().trim(),
                entityType = dialogBinding.etEntityType.text.toString().trim()
            )
            updateManagementEntity(updatedEntity)
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun generateNextMgmtId(entities: List<ManagementEntity>): String {
        if (entities.isEmpty()) return "MG000001"

        val maxId = entities.mapNotNull { entity ->
            entity.mgmtId.removePrefix("MG").toIntOrNull()
        }.maxOrNull() ?: 0

        return "MG" + String.format("%06d", maxId + 1)
    }

    private fun addManagementEntity(entity: ManagementEntity) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.addManagementEntity(entity)
                if (response.isSuccessful) {
                    showSuccess("Management entity added successfully")
                    loadData()
                } else {
                    showError("Failed to add: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun updateManagementEntity(entity: ManagementEntity) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.updateManagementEntity(entity.mgmtId, entity)
                if (response.isSuccessful) {
                    showSuccess("Management entity updated successfully")
                    loadData()
                } else {
                    showError("Failed to update: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun navigateToLegalEntities(leId: String) {
        val fragment = LegalEntitiesFragment()
        val bundle = Bundle().apply {
            putString("search_query", leId)
            putString("filter_field", "LE_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("legal", bundle)
    }

    private fun navigateToFunds(mgmtId: String) {
        val fragment = FundsFragment()
        val bundle = Bundle().apply {
            putString("search_query", mgmtId)
            putString("filter_field", "MGMT_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("funds", bundle)
    }

    private fun navigateToSubFunds(mgmtId: String) {
        val fragment = SubFundsFragment()
        val bundle = Bundle().apply {
            putString("search_query", mgmtId)
            putString("filter_field", "MGMT_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("subfunds", bundle)
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