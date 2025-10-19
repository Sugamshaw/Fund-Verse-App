package adapters

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.fundbank.databinding.ItemFundBinding
import models.FundMaster


class FundAdapter(private val onEditClick: (FundMaster) -> Unit,
                  private val onSubFundsClick: (String) -> Unit,
                  private val onShareclassClick: (String) -> Unit,
                  private val onLeIdClick: (String) -> Unit,
                  private val onMgmtIdClick:(String) -> Unit) : ListAdapter<FundMaster, FundAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = ViewHolder(
        ItemFundBinding.inflate(LayoutInflater.from(parent.context), parent, false)
    )

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(private val binding: ItemFundBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(fund: FundMaster) {
            binding.apply {
                tvFundId.text = fund.fundId
                tvFundCode.text = "Code: ${fund.fundCode}"
                tvFundName.text = fund.fundName
                tvFundType.text = fund.fundType
                tvCurrency.text = "Currency: "+fund.baseCurrency
                tvDomicile.text = fund.domicile
                tvLeId.text = fund.leId
                tvMgmtId.text = fund.mgmtId
                tvStatus.apply {
                    text = fund.status
                    setTextColor(if (fund.status == "ACTIVE") Color.parseColor("#166534") else Color.parseColor("#6B7280"))
                    setBackgroundColor(if (fund.status == "ACTIVE") Color.parseColor("#DCFCE7") else Color.parseColor("#F3F4F6"))
                }
                btnEdit.setOnClickListener {onEditClick(fund) }
                btnSubfunds.setOnClickListener { onSubFundsClick(fund.fundId) }
                btnShareclass.setOnClickListener { onShareclassClick(fund.fundId) }
                btnTvLeId.setOnClickListener { onLeIdClick(fund.leId) }
                btnTvMgmtId.setOnClickListener { onMgmtIdClick(fund.mgmtId) }
            }
        }
    }

    class DiffCallback : DiffUtil.ItemCallback<FundMaster>() {
        override fun areItemsTheSame(oldItem: FundMaster, newItem: FundMaster) = oldItem.fundId == newItem.fundId
        override fun areContentsTheSame(oldItem: FundMaster, newItem: FundMaster) = oldItem == newItem
    }
}