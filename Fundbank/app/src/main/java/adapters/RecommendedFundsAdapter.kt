package adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.example.fundbank.R
import models.AIRecommendation

import java.text.NumberFormat
import java.util.*

/**
 * Adapter for displaying AI-recommended funds in RecyclerView
 */
class RecommendedFundsAdapter(
    private var recommendations: List<AIRecommendation>,
    private val onFundClick: (AIRecommendation) -> Unit
) : RecyclerView.Adapter<RecommendedFundsAdapter.RecommendationViewHolder>() {

    inner class RecommendationViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val cardFund: CardView = itemView.findViewById(R.id.card_fund)
        private val tvRank: TextView = itemView.findViewById(R.id.tv_rank)
        private val tvRecommendationBadge: TextView = itemView.findViewById(R.id.tv_recommendation_badge)
        private val tvFundName: TextView = itemView.findViewById(R.id.tv_fund_name)
        private val tvFundId: TextView = itemView.findViewById(R.id.tv_fund_id)
        private val tvInvestmentScore: TextView = itemView.findViewById(R.id.tv_investment_score)
        private val tvScoreRating: TextView = itemView.findViewById(R.id.tv_score_rating)
        private val tvNav: TextView = itemView.findViewById(R.id.tv_nav)
        private val tvAum: TextView = itemView.findViewById(R.id.tv_aum)
        private val tvExpenseRatio: TextView = itemView.findViewById(R.id.tv_expense_ratio)
        private val tvPerformanceClass: TextView = itemView.findViewById(R.id.tv_performance_class)
        private val tvRiskLevel: TextView = itemView.findViewById(R.id.tv_risk_level)
        private val tvRecommendation: TextView = itemView.findViewById(R.id.tv_recommendation)
        private val btnViewDetails: View = itemView.findViewById(R.id.btn_view_details)

        fun bind(recommendation: AIRecommendation, position: Int) {
            // Set rank
            tvRank.text = (position + 1).toString()

            // Change rank badge color for top 3
            when (position) {
                0 -> tvRank.setBackgroundResource(R.drawable.circle_background_gold)
                1 -> tvRank.setBackgroundResource(R.drawable.circle_background_silver)
                2 -> tvRank.setBackgroundResource(R.drawable.circle_background_bronze)
                else -> tvRank.setBackgroundResource(R.drawable.circle_background_primary)
            }

            // Basic info
            tvFundName.text = recommendation.fundName
            tvFundId.text = recommendation.fundId

            // Investment score
            tvInvestmentScore.text = String.format("%.2f", recommendation.investmentScore)
            tvScoreRating.text = getStarRating(recommendation.investmentScore)

            // Financial metrics
            tvNav.text = formatCurrency(recommendation.nav)
            tvAum.text = formatLargeNumber(recommendation.aum)
            tvExpenseRatio.text = String.format("%.2f%%", recommendation.expenseRatio * 100)

            // Performance class
            tvPerformanceClass.text = recommendation.performanceClass
            when (recommendation.performanceClass) {
                "HIGH" -> tvPerformanceClass.setBackgroundResource(R.drawable.rounded_background_success)
                "MEDIUM" -> tvPerformanceClass.setBackgroundResource(R.drawable.rounded_background_warning)
                "LOW" -> tvPerformanceClass.setBackgroundResource(R.drawable.rounded_background_error)
            }

            // Risk level
            tvRiskLevel.text = recommendation.riskLevel
            when (recommendation.riskLevel) {
                "LOW" -> tvRiskLevel.setBackgroundResource(R.drawable.rounded_background_success)
                "MEDIUM" -> tvRiskLevel.setBackgroundResource(R.drawable.rounded_background_warning)
                "HIGH" -> tvRiskLevel.setBackgroundResource(R.drawable.rounded_background_error)
            }

            // Recommendation
            tvRecommendation.text = recommendation.recommendation
            tvRecommendationBadge.text = recommendation.recommendation

            val (badgeColor, textColor) = when (recommendation.recommendation) {
                "BUY" -> R.drawable.rounded_background_success to R.color.success_green
                "HOLD" -> R.drawable.rounded_background_warning to R.color.warning_yellow
                "AVOID" -> R.drawable.rounded_background_error to R.color.error_red
                else -> R.drawable.rounded_background_success to R.color.success_green
            }

            tvRecommendationBadge.setBackgroundResource(badgeColor)
            tvRecommendation.setTextColor(ContextCompat.getColor(itemView.context, textColor))

            // Click listeners
            cardFund.setOnClickListener { onFundClick(recommendation) }
            btnViewDetails.setOnClickListener { onFundClick(recommendation) }
        }

        private fun getStarRating(score: Double): String {
            val stars = when {
                score >= 9.0 -> "⭐⭐⭐⭐⭐"
                score >= 7.5 -> "⭐⭐⭐⭐"
                score >= 6.0 -> "⭐⭐⭐"
                score >= 4.0 -> "⭐⭐"
                else -> "⭐"
            }
            return stars
        }

        private fun formatCurrency(value: Double): String {
            val formatter = NumberFormat.getCurrencyInstance(Locale.US)
            return formatter.format(value)
        }

        private fun formatLargeNumber(value: Double): String {
            return when {
                value >= 1_000_000_000_000 -> String.format("$%.1fT", value / 1_000_000_000_000)
                value >= 1_000_000_000 -> String.format("$%.1fB", value / 1_000_000_000)
                value >= 1_000_000 -> String.format("$%.1fM", value / 1_000_000)
                else -> formatCurrency(value)
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecommendationViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_recommended_fund, parent, false)
        return RecommendationViewHolder(view)
    }

    override fun onBindViewHolder(holder: RecommendationViewHolder, position: Int) {
        holder.bind(recommendations[position], position)
    }

    override fun getItemCount(): Int = recommendations.size

    /**
     * Update the list of recommendations
     */
    fun updateData(newRecommendations: List<AIRecommendation>) {
        recommendations = newRecommendations
        notifyDataSetChanged()
    }
}