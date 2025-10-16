package fragments

import adapters.LegalEntityAdapter
import android.app.Dialog
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import api.RetrofitClient
import com.example.fundbank.MainActivity
import com.example.fundbank.R
import com.example.fundbank.RefreshableFragment
import com.example.fundbank.databinding.FragmentDialogLegalEntityBinding
import com.example.fundbank.databinding.FragmentLegalEntitiesBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.launch
import models.LegalEntity

class LegalEntitiesFragment : Fragment(), RefreshableFragment {
    private var _binding: FragmentLegalEntitiesBinding? = null
    private val binding get() = _binding!!

    private lateinit var adapter: LegalEntityAdapter
    private var allEntities: List<LegalEntity> = emptyList()
    private var searchQuery: String? = null
    private var filterField: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLegalEntitiesBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.progressBar.indeterminateTintList =
            ColorStateList.valueOf(ContextCompat.getColor(requireContext(), R.color.blue_600))

        setupRecyclerView()
        setupSearch()
        setupSwipeRefresh()
        setupFab()

        searchQuery = arguments?.getString("search_query")
        filterField = arguments?.getString("filter_field")

        loadData()
    }

    private fun setupRecyclerView() {
        adapter = LegalEntityAdapter(
            onEditClick = { entity -> showEditDialog(entity) },
            onManagementClick = { leId -> navigateToManagement(leId) },
            onFundsClick = { leId -> navigateToFunds(leId) },
            onSubFundsClick = { leId -> navigateToSubFunds(leId) }
        )
        binding.recyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@LegalEntitiesFragment.adapter
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
        binding.fabAdd.setOnClickListener {
            showAddDialog()
        }

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
        val filtered = if (query.isEmpty()) {
            allEntities
        } else {
            allEntities.filter {
                it.legalName.contains(query, ignoreCase = true) ||
                        it.leId.contains(query, ignoreCase = true) ||
                        it.jurisdiction.contains(query, ignoreCase = true) ||
                        it.entityType.contains(query, ignoreCase = true) ||
                        it.lei.contains(query, ignoreCase = true)
            }
        }
        adapter.submitList(filtered)
        updateEmptyState(filtered.isEmpty())
    }

    private fun showAddDialog() {
        val dialogBinding = FragmentDialogLegalEntityBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        dialogBinding.tvDialogTitle.text = "Add Legal Entity"

        dialogBinding.etLeId.isEnabled = false

        val nextLeId = generateNextLeId(allEntities)
        dialogBinding.etLeId.setText(nextLeId)

        dialogBinding.btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        dialogBinding.btnSubmit.setOnClickListener {
            if (validateInputFields(dialogBinding, isEdit = false)) {
                val entity = LegalEntity(
                    leId = nextLeId,
                    lei = dialogBinding.etLei.text.toString().trim(),
                    legalName = dialogBinding.etLegalName.text.toString().trim(),
                    jurisdiction = dialogBinding.etJurisdiction.text.toString().trim(),
                    entityType = dialogBinding.etEntityType.text.toString().trim()
                )
                addLegalEntity(entity)
                dialog.dismiss()
            }
        }

        dialog.show()
    }

    private fun generateNextLeId(entities: List<LegalEntity>): String {
        if (entities.isEmpty()) return "LE000001"

        val maxId = entities.mapNotNull {
            it.leId.removePrefix("LE").toIntOrNull()
        }.maxOrNull() ?: 0

        val nextId = maxId + 1
        return "LE" + String.format("%06d", nextId)
    }

    private fun showEditDialog(entity: LegalEntity) {
        val dialogBinding = FragmentDialogLegalEntityBinding.inflate(layoutInflater)
        val dialog = Dialog(requireContext())
        dialog.setContentView(dialogBinding.root)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        dialogBinding.tvDialogTitle.text = "Edit Legal Entity"

        dialogBinding.apply {
            etLeId.setText(entity.leId)
            etLeId.isEnabled = false
            etLei.setText(entity.lei)
            etLegalName.setText(entity.legalName)
            etJurisdiction.setText(entity.jurisdiction)
            etEntityType.setText(entity.entityType)
        }

        dialogBinding.btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        dialogBinding.btnSubmit.text = "Update"
        dialogBinding.btnSubmit.setOnClickListener {
            if (validateInputFields(dialogBinding, isEdit = true)) {
                val updatedEntity = LegalEntity(
                    leId = entity.leId,
                    lei = dialogBinding.etLei.text.toString().trim(),
                    legalName = dialogBinding.etLegalName.text.toString().trim(),
                    jurisdiction = dialogBinding.etJurisdiction.text.toString().trim(),
                    entityType = dialogBinding.etEntityType.text.toString().trim()
                )
                updateLegalEntity(updatedEntity)
                dialog.dismiss()
            }
        }

        dialog.show()
    }

    private fun validateInputFields(
        binding: FragmentDialogLegalEntityBinding,
        isEdit: Boolean = false
    ): Boolean {
        var isValid = true

        binding.tilLei.error = null
        binding.tilLegalName.error = null
        binding.tilJurisdiction.error = null
        binding.tilEntityType.error = null

        if (binding.etLei.text.isNullOrBlank()) {
            binding.tilLei.error = "LEI is required"
            isValid = false
        }

        if (binding.etLegalName.text.isNullOrBlank()) {
            binding.tilLegalName.error = "Legal Name is required"
            isValid = false
        }

        if (binding.etJurisdiction.text.isNullOrBlank()) {
            binding.tilJurisdiction.error = "Jurisdiction is required"
            isValid = false
        }

        if (binding.etEntityType.text.isNullOrBlank()) {
            binding.tilEntityType.error = "Entity Type is required"
            isValid = false
        }

        return isValid
    }

    private fun addLegalEntity(entity: LegalEntity) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.addLegalEntity(entity)
                if (response.isSuccessful) {
                    showSuccess("Legal entity added successfully")
                    loadData()
                } else {
                    showError("Failed to add: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun updateLegalEntity(entity: LegalEntity) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.updateLegalEntity(entity.leId, entity)
                if (response.isSuccessful) {
                    showSuccess("Legal entity updated successfully")
                    loadData()
                } else {
                    showError("Failed to update: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun deleteLegalEntity(leId: String) {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.deleteLegalEntity(leId)
                if (response.isSuccessful) {
                    showSuccess("Legal entity deleted successfully")
                    loadData()
                } else {
                    showError("Failed to delete: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun navigateToManagement(leId: String) {
        val fragment = ManagementEntitiesFragment()
        val bundle = Bundle().apply {
            putString("search_query", leId)
            putString("filter_field", "LE_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("management", bundle)
    }

    private fun navigateToFunds(leId: String) {
        val fragment = FundsFragment()
        val bundle = Bundle().apply {
            putString("search_query", leId)
            putString("filter_field", "LE_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("funds", bundle)
    }

    private fun navigateToSubFunds(leId: String) {
        val fragment = SubFundsFragment()
        val bundle = Bundle().apply {
            putString("search_query", leId)
            putString("filter_field", "LE_ID")
        }
        (requireActivity() as? MainActivity)?.openFragmentFromChild("subfunds", bundle)
    }

    private fun loadData() {
        if (!isAdded || _binding == null) return
        binding.swipeRefresh.isRefreshing = true

        lifecycleScope.launch {
            try {
                val response = RetrofitClient.apiService.getLegalEntities()
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
                        binding.filterInfo.text = "Filtered by: LE_ID = $searchQuery"

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

    private fun updateEmptyState(isEmpty: Boolean) {
        binding.emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
        binding.recyclerView.visibility = if (isEmpty) View.GONE else View.VISIBLE
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Retry") { loadData() }
            .show()
    }

    private fun showSuccess(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT)
            .setBackgroundTint(Color.parseColor("#4CAF50"))
            .setTextColor(Color.WHITE)
            .show()
    }

    override fun refresh() {
        loadData()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}