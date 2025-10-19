package adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.fundbank.databinding.ItemManagementEntityBinding
import models.ManagementEntity

class ManagementEntityAdapter(
    private val onEditClick: (ManagementEntity) -> Unit,
    private val onFundsClick: (String) -> Unit,
    private val onSubFundsClick: (String) -> Unit,
    private val onLeIdClick: (String) -> Unit   // ✅ New callback for LE ID click
) : ListAdapter<ManagementEntity, ManagementEntityAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = ViewHolder(
        ItemManagementEntityBinding.inflate(LayoutInflater.from(parent.context), parent, false)
    )

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(private val binding: ItemManagementEntityBinding) :
        RecyclerView.ViewHolder(binding.root) {
        fun bind(entity: ManagementEntity) {
            binding.apply {
                tvMgmtId.text = entity.mgmtId
                tvLeId.text = entity.leId
                tvRegistration.text = entity.registrationNo
                tvDomicile.text = entity.domicile
                tvEntityType.text = entity.entityType

                // ✅ Click Listeners
                btnEdit.setOnClickListener { onEditClick(entity) }
                btnFunds.setOnClickListener { onFundsClick(entity.mgmtId) }
                btnSubfunds.setOnClickListener { onSubFundsClick(entity.mgmtId) }
                bgTvLeId.setOnClickListener { onLeIdClick(entity.leId) } // 👈 Navigate to Legal Entities
            }
        }
    }

    class DiffCallback : DiffUtil.ItemCallback<ManagementEntity>() {
        override fun areItemsTheSame(oldItem: ManagementEntity, newItem: ManagementEntity) =
            oldItem.mgmtId == newItem.mgmtId

        override fun areContentsTheSame(oldItem: ManagementEntity, newItem: ManagementEntity) =
            oldItem == newItem
    }
}
