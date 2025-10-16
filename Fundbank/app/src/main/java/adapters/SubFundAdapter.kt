package adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.fundbank.databinding.ItemSubFundBinding
import models.SubFund

class SubFundAdapter(
    private val onEditClick: (SubFund) -> Unit,
    private val onParentFundClick: (String) -> Unit,
    private val onMgmtIdClick: (String) -> Unit,
    private val onLeIdClick: (String) -> Unit,
    private val onShareClassClick: (String) -> Unit
) : ListAdapter<SubFund, SubFundAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = ViewHolder(
        ItemSubFundBinding.inflate(LayoutInflater.from(parent.context), parent, false)
    )

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(private val binding: ItemSubFundBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(subFund: SubFund) {
            binding.apply {
                tvSubfundId.text = subFund.subfundId
                tvParentFundId.text = subFund.parentFundId
                tvMgmtId.text = subFund.mgmtId
                tvLeId.text = subFund.leId
                tvIsin.text = "ISIN: ${subFund.isinSub}"
                tvCurrency.text = "Currency: ${subFund.currency}"


                btnEdit.setOnClickListener { onEditClick(subFund) }
                btnTvParentFundId.setOnClickListener { onParentFundClick(subFund.parentFundId) }
                btnTvMgmtId.setOnClickListener { onMgmtIdClick(subFund.mgmtId) }
                btnTvLeId.setOnClickListener { onLeIdClick(subFund.leId) }
                btnShareclass.setOnClickListener { onShareClassClick(subFund.parentFundId) }
            }
        }
    }

    class DiffCallback : DiffUtil.ItemCallback<SubFund>() {
        override fun areItemsTheSame(oldItem: SubFund, newItem: SubFund) =
            oldItem.subfundId == newItem.subfundId

        override fun areContentsTheSame(oldItem: SubFund, newItem: SubFund) =
            oldItem == newItem
    }
}
