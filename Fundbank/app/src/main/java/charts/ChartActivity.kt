package charts

import android.graphics.Color
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.fundbank.databinding.ActivityChartBinding
import api.RetrofitClient
import models.LegalEntity
import models.FundMaster
import models.ShareClass
import com.github.mikephil.charting.animation.Easing
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import com.github.mikephil.charting.formatter.PercentFormatter
import com.github.mikephil.charting.utils.ColorTemplate
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.launch

class ChartActivity : AppCompatActivity() {
    private lateinit var binding: ActivityChartBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChartBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Analytics Dashboard"
        binding.toolbar.navigationIcon?.setTint(Color.WHITE)
        loadChartData()
    }

    private fun loadChartData() {
        binding.progressBar.visibility = View.VISIBLE

        lifecycleScope.launch {
            try {
                val legalEntities = RetrofitClient.apiService.getLegalEntities().body() ?: emptyList()
                val funds = RetrofitClient.apiService.getFunds().body() ?: emptyList()
                val shareClasses = RetrofitClient.apiService.getShareClasses().body() ?: emptyList()

                setupEntityTypesPieChart(legalEntities)
                setupFundStatusBarChart(funds)
                setupShareClassAUMChart(shareClasses.take(10))
                setupCurrencyDistribution(shareClasses)

                binding.progressBar.visibility = View.GONE
                binding.scrollView.visibility = View.VISIBLE

            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                Snackbar.make(binding.root, "Error loading charts: ${e.message}",
                    Snackbar.LENGTH_LONG).show()
            }
        }
    }

    private fun setupEntityTypesPieChart(entities: List<LegalEntity>) {
        val pieChart = binding.pieChartEntityTypes
        val typeCounts = entities.groupingBy { it.entityType }.eachCount()
        val entries = typeCounts.map { PieEntry(it.value.toFloat(), it.key) }

        val dataSet = PieDataSet(entries, "Entity Types").apply {
            colors = ColorTemplate.MATERIAL_COLORS.toList()
            valueTextSize = 14f
            valueTextColor = Color.WHITE
            sliceSpace = 3f
            selectionShift = 5f
        }

        pieChart.apply {
            data = PieData(dataSet).apply {
                setValueFormatter(PercentFormatter(pieChart))
            }
            description.isEnabled = false
            setUsePercentValues(true)
            setEntryLabelColor(Color.BLACK)
            setEntryLabelTextSize(12f)
            centerText = "Entity Types\n${entities.size} Total"
            setCenterTextSize(14f)
            isDrawHoleEnabled = true
            setHoleColor(Color.WHITE)
            holeRadius = 45f
            transparentCircleRadius = 50f
            legend.apply {
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
                textSize = 12f
            }
            animateY(1400, Easing.EaseInOutQuad)
            invalidate()
        }
    }

    private fun setupFundStatusBarChart(funds: List<FundMaster>) {
        val barChart = binding.barChartFundStatus
        val statusCounts = funds.groupingBy { it.status }.eachCount()
        val entries = statusCounts.entries.mapIndexed { index, entry ->
            BarEntry(index.toFloat(), entry.value.toFloat())
        }
        val labels = statusCounts.keys.toList()

        val dataSet = BarDataSet(entries, "Fund Status").apply {
            colors = listOf(Color.parseColor("#16A34A"), Color.parseColor("#6B7280"))
            valueTextSize = 14f
            valueTextColor = Color.BLACK
        }

        barChart.apply {
            data = BarData(dataSet).apply { barWidth = 0.8f }
            description.text = "Funds by Status"
            description.textSize = 14f
            xAxis.apply {
                valueFormatter = IndexAxisValueFormatter(labels)
                position = XAxis.XAxisPosition.BOTTOM
                granularity = 1f
                textSize = 12f
                setDrawGridLines(false)
            }
            axisLeft.apply {
                textSize = 12f
                axisMinimum = 0f
                granularity = 1f
            }
            axisRight.isEnabled = false
            legend.apply {
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                horizontalAlignment = Legend.LegendHorizontalAlignment.RIGHT
                orientation = Legend.LegendOrientation.VERTICAL
                setDrawInside(true)
                textSize = 12f
            }
            animateY(1400)
            invalidate()
        }
    }

    private fun setupShareClassAUMChart(shareClasses: List<ShareClass>) {
        val barChart = binding.barChartAum
        val entries = shareClasses.mapIndexed { index, sc ->
            BarEntry(index.toFloat(), (sc.aum / 1_000_000).toFloat())
        }
        val labels = shareClasses.map { it.scId }

        val dataSet = BarDataSet(entries, "AUM (Millions)").apply {
            color = Color.parseColor("#2563EB")
            valueTextSize = 10f
            valueTextColor = Color.BLACK
        }

        barChart.apply {
            data = BarData(dataSet).apply { barWidth = 0.9f }
            description.text = "Top 10 Share Classes by AUM"
            description.textSize = 14f
            xAxis.apply {
                valueFormatter = IndexAxisValueFormatter(labels)
                position = XAxis.XAxisPosition.BOTTOM
                granularity = 1f
                textSize = 10f
                setDrawGridLines(false)
                labelRotationAngle = -45f
            }
            axisLeft.apply {
                textSize = 12f
                axisMinimum = 0f
            }
            axisRight.isEnabled = false
            legend.apply {
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                textSize = 12f
            }
            animateY(1400)
            invalidate()
        }
    }

    private fun setupCurrencyDistribution(shareClasses: List<ShareClass>) {
        val pieChart = binding.pieChartCurrency
        val currencyCounts = shareClasses.groupingBy { it.currency }.eachCount()
        val entries = currencyCounts.map { PieEntry(it.value.toFloat(), it.key) }

        val dataSet = PieDataSet(entries, "Currency Distribution").apply {
            colors = ColorTemplate.COLORFUL_COLORS.toList()
            valueTextSize = 14f
            valueTextColor = Color.WHITE
            sliceSpace = 2f
        }

        pieChart.apply {
            data = PieData(dataSet).apply {
                setValueFormatter(PercentFormatter(pieChart))
            }
            description.isEnabled = false
            setUsePercentValues(true)
            setEntryLabelColor(Color.BLACK)
            setEntryLabelTextSize(11f)
            centerText = "Currencies\n${shareClasses.size} Classes"
            setCenterTextSize(14f)
            isDrawHoleEnabled = true
            setHoleColor(Color.WHITE)
            holeRadius = 40f
            legend.apply {
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
                isWordWrapEnabled = true
                textSize = 11f
            }
            animateY(1400, Easing.EaseInOutQuad)
            invalidate()
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }
}