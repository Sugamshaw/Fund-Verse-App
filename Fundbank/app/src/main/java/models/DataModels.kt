package models

import android.annotation.SuppressLint
import com.google.gson.annotations.SerializedName
import android.os.Parcelable
import kotlinx.parcelize.Parcelize


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