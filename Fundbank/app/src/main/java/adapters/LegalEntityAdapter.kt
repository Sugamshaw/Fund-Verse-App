package adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.fundbank.databinding.ItemLegalEntityBinding
import models.LegalEntity

class LegalEntityAdapter(
    private val onEditClick: (LegalEntity) -> Unit,
    private val onManagementClick: (String) -> Unit,
    private val onFundsClick: (String) -> Unit,
    private val onSubFundsClick: (String) -> Unit
) : ListAdapter<LegalEntity, LegalEntityAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemLegalEntityBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(
        private val binding: ItemLegalEntityBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(entity: LegalEntity) {
            binding.apply {
                tvLeId.text = entity.leId
                tvLegalName.text = entity.legalName
                tvLei.text = entity.lei
                tvJurisdiction.text = entity.jurisdiction
                tvEntityType.text = entity.entityType

                btnEdit.setOnClickListener { onEditClick(entity) }
                btnManagement.setOnClickListener { onManagementClick(entity.leId) }
                btnFunds.setOnClickListener { onFundsClick(entity.leId) }
                btnSubfunds.setOnClickListener { onSubFundsClick(entity.leId) }
            }
        }
    }

    private class DiffCallback : DiffUtil.ItemCallback<LegalEntity>() {
        override fun areItemsTheSame(oldItem: LegalEntity, newItem: LegalEntity) =
            oldItem.leId == newItem.leId

        override fun areContentsTheSame(oldItem: LegalEntity, newItem: LegalEntity) =
            oldItem == newItem
    }
}
