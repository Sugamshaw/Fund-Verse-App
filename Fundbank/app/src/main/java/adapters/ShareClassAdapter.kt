package adapters

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.fundbank.databinding.ItemShareClassBinding
import models.ShareClass
import java.text.DecimalFormat

class ShareClassAdapter(
    private val onEditClick: (ShareClass) -> Unit,
    private val onFundIdClick: (String) -> Unit
) : ListAdapter<ShareClass, ShareClassAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = ViewHolder(
        ItemShareClassBinding.inflate(LayoutInflater.from(parent.context), parent, false)
    )

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(private val binding: ItemShareClassBinding) :
        RecyclerView.ViewHolder(binding.root) {

        private val currencyFormatter = DecimalFormat("#,##0.00")
        private val percentFormatter = DecimalFormat("0.00")

        fun bind(shareClass: ShareClass) = with(binding) {

            // 🔹 Basic Details
            tvScId.text = shareClass.scId
            tvFundId.text = "Fund ID: ${shareClass.fundId}"
            tvIsinSc.text = "ISIN: ${shareClass.isinSc}"
            tvCurrency.text = "Currency: ${shareClass.currency}"
            tvDistribution.text = "Distribution: ${shareClass.distribution}"

            // 🔹 Fee Details
            tvFeeMgmt.text = "${percentFormatter.format(shareClass.feeMgmt )}%"
            tvPerfFee.text = "${percentFormatter.format(shareClass.perfFee )}%"
            tvExpenseRatio.text = "${percentFormatter.format(shareClass.expenseRatio )}%"

            // 🔹 Financial Data
            tvNav.text = "$${currencyFormatter.format(shareClass.nav)}"
            tvAum.text = "$${currencyFormatter.format(shareClass.aum / 1_000_000)}M"

            // 🔹 Status Styling
            tvStatus.apply {
                text = shareClass.status
                val isActive = shareClass.status.equals("ACTIVE", ignoreCase = true)
                setTextColor(if (isActive) Color.parseColor("#166534") else Color.parseColor("#6B7280"))
                setBackgroundColor(if (isActive) Color.parseColor("#DCFCE7") else Color.parseColor("#F3F4F6"))
            }

            // 🔹 Click Listeners
            btnEdit.setOnClickListener { onEditClick(shareClass) }
            btnTvFundId.setOnClickListener { onFundIdClick(shareClass.fundId) }
        }
    }

    class DiffCallback : DiffUtil.ItemCallback<ShareClass>() {
        override fun areItemsTheSame(oldItem: ShareClass, newItem: ShareClass) =
            oldItem.scId == newItem.scId

        override fun areContentsTheSame(oldItem: ShareClass, newItem: ShareClass) =
            oldItem == newItem
    }
}
