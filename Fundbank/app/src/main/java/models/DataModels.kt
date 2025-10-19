package models

import android.annotation.SuppressLint
import com.google.gson.annotations.SerializedName
import android.os.Parcelable
import kotlinx.parcelize.Parcelize


/**
 * AI Performance Statistics
 */
data class PerformanceStats(
    @SerializedName("model_accuracy")
    val modelAccuracy: Double,

    @SerializedName("total_funds")
    val totalFunds: Int,

    @SerializedName("high_performers")
    val highPerformers: Int,

    @SerializedName("medium_performers")
    val mediumPerformers: Int,

    @SerializedName("low_performers")
    val lowPerformers: Int,

    @SerializedName("high_risk_funds")
    val highRiskFunds: Int,

    @SerializedName("medium_risk_funds")
    val mediumRiskFunds: Int,

    @SerializedName("low_risk_funds")
    val lowRiskFunds: Int
)

/**
 * AI Fund Recommendation
 */
data class FeatureImportance(
    val features: List<Feature>
)

data class Feature(
    val name: String,
    val importance: Double,
    val rank: Int
)
data class AIRecommendation(
    @SerializedName("fund_id")
    val fundId: String,

    @SerializedName("fund_name")
    val fundName: String,

    @SerializedName("nav")
    val nav: Double,

    @SerializedName("aum")
    val aum: Double,

    @SerializedName("investment_score")
    val investmentScore: Double,

    @SerializedName("performance_class")
    val performanceClass: String, // HIGH, MEDIUM, LOW

    @SerializedName("risk_level")
    val riskLevel: String, // HIGH, MEDIUM, LOW

    @SerializedName("recommendation")
    val recommendation: String, // BUY, HOLD, AVOID

    @SerializedName("expense_ratio")
    val expenseRatio: Double
)

/**
 * NAV Prediction
 */
data class NAVPrediction(
    @SerializedName("fund_id")
    val fundId: String,

    @SerializedName("fund_name")
    val fundName: String,

    @SerializedName("current_nav")
    val currentNAV: Double,

    @SerializedName("predicted_nav")
    val predictedNAV: Double,

    @SerializedName("confidence")
    val confidence: Double,

    @SerializedName("prediction_date")
    val predictionDate: String,

    @SerializedName("change_percentage")
    val changePercentage: Double
)

/**
 * Performance Classification Response
 */
data class PerformanceClassification(
    @SerializedName("fund_id")
    val fundId: String,

    @SerializedName("fund_name")
    val fundName: String,

    @SerializedName("performance_class")
    val performanceClass: String, // HIGH, MEDIUM, LOW

    @SerializedName("confidence")
    val confidence: Double,

    @SerializedName("recommendation")
    val recommendation: String // BUY, HOLD, AVOID
)

/**
 * Risk Assessment Response
 */
data class RiskAssessment(
    @SerializedName("high_risk_count")
    val highRiskCount: Int,

    @SerializedName("avg_high_risk_expense")
    val avgHighRiskExpense: Double,

    @SerializedName("avg_high_risk_perf_fee")
    val avgHighRiskPerfFee: Double,

    @SerializedName("recommendation")
    val recommendation: String,

    @SerializedName("high_risk_funds")
    val highRiskFunds: List<String>
)

/**
 * Fund Risk Detail
 */
data class FundRiskDetail(
    @SerializedName("fund_id")
    val fundId: String,

    @SerializedName("fund_name")
    val fundName: String,

    @SerializedName("risk_level")
    val riskLevel: String,

    @SerializedName("risk_score")
    val riskScore: Double,

    @SerializedName("expense_ratio")
    val expenseRatio: Double,

    @SerializedName("perf_fee")
    val perfFee: Double,

    @SerializedName("warning_message")
    val warningMessage: String?
)

@Parcelize
data class LegalEntity(
    @SerializedName("LE_ID") val leId: String,
    @SerializedName("LEI") val lei: String,
    @SerializedName("LEGAL_NAME") val legalName: String,
    @SerializedName("JURISDICTION") val jurisdiction: String,
    @SerializedName("ENTITY_TYPE") val entityType: String
) : Parcelable

@Parcelize
data class ManagementEntity(
    @SerializedName("MGMT_ID") val mgmtId: String,
    @SerializedName("LE_ID") val leId: String,
    @SerializedName("REGISTRATION_NO") val registrationNo: String,
    @SerializedName("DOMICILE") val domicile: String,
    @SerializedName("ENTITY_TYPE") val entityType: String
) : Parcelable

@Parcelize
data class FundMaster(
    @SerializedName("FUND_ID") val fundId: String,
    @SerializedName("MGMT_ID") val mgmtId: String,
    @SerializedName("LE_ID") val leId: String,
    @SerializedName("FUND_CODE") val fundCode: String,
    @SerializedName("FUND_NAME") val fundName: String,
    @SerializedName("FUND_TYPE") val fundType: String,
    @SerializedName("BASE_CURRENCY") val baseCurrency: String,
    @SerializedName("DOMICILE") val domicile: String,
    @SerializedName("ISIN_MASTER") val isinMaster: String,
    @SerializedName("STATUS") val status: String
) : Parcelable


@Parcelize
data class SubFund(
    @SerializedName("SUBFUND_ID") val subfundId: String,
    @SerializedName("PARENT_FUND_ID") val parentFundId: String,
    @SerializedName("LE_ID") val leId: String,
    @SerializedName("MGMT_ID") val mgmtId: String,
    @SerializedName("ISIN_SUB") val isinSub: String,
    @SerializedName("CURRENCY") val currency: String
) : Parcelable

@Parcelize
data class ShareClass(
    @SerializedName("SC_ID") val scId: String,
    @SerializedName("FUND_ID") val fundId: String,
    @SerializedName("ISIN_SC") val isinSc: String,
    @SerializedName("CURRENCY") val currency: String,
    @SerializedName("DISTRIBUTION") val distribution: String,
    @SerializedName("FEE_MGMT") val feeMgmt: Double,
    @SerializedName("PERF_FEE") val perfFee: Double,
    @SerializedName("EXPENSE_RATIO") val expenseRatio: Double,
    @SerializedName("NAV") val nav: Double,
    @SerializedName("AUM") val aum: Double,
    @SerializedName("STATUS") val status: String
) : Parcelable

data class ApiResponse<T>(
    val data: T? = null,
    val error: String? = null
)