package com.example.fundbank

import adapters.RecommendedFundsAdapter
import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import api.RetrofitClient
import com.example.fundbank.databinding.ActivityAllnsightsBinding
import com.github.mikephil.charting.animation.Easing
import com.github.mikephil.charting.charts.BarChart
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import com.github.mikephil.charting.formatter.PercentFormatter
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.tabs.TabLayout
import kotlinx.coroutines.launch
import models.AIRecommendation
import models.PerformanceStats
import models.NAVPrediction
import models.RiskAssessment
import models.PerformanceClassification

class AIInsightsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAllnsightsBinding
    private lateinit var recommendationsAdapter: RecommendedFundsAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAllnsightsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupTabs()
        setupRecyclerView()
        setupSearchFunctionality()
        loadAIInsights()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setDisplayShowHomeEnabled(true)
            title = "AI Insights"
        }

        binding.toolbar.setNavigationOnClickListener {
            onBackPressed()
        }
    }

    private fun setupTabs() {
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("📊 Overview"))
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("🎯 Predictions"))
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("💡 Recommendations"))
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("⚠️ Risk Analysis"))

        binding.tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                when (tab?.position) {
                    0 -> showOverview()
                    1 -> showPredictions()
                    2 -> showRecommendations()
                    3 -> showRiskAnalysis()
                }
            }
            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })
    }

    private fun setupRecyclerView() {
        recommendationsAdapter = RecommendedFundsAdapter(emptyList()) { fund ->
            // Handle fund click
            Snackbar.make(
                binding.root,
                "Opening ${fund.fundName}...",
                Snackbar.LENGTH_SHORT
            ).show()
        }

        binding.recyclerViewRecommendations.apply {
            layoutManager = LinearLayoutManager(this@AIInsightsActivity)
            adapter = recommendationsAdapter
        }
    }

    /**
     * 🔍 Setup Search Functionality
     */
    private fun setupSearchFunctionality() {
        // Search button click listener
        binding.btnSearch.setOnClickListener {
            performSearch()
        }

        // Handle Enter key on keyboard
        binding.searchEditText.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                performSearch()
                true
            } else {
                false
            }
        }
    }

    /**
     * 🔍 Perform Fund Search
     */
    private fun performSearch() {
        val fundId = binding.searchEditText.text.toString().trim()

        if (fundId.isEmpty()) {
            Snackbar.make(
                binding.root,
                "Please enter a Fund ID",
                Snackbar.LENGTH_SHORT
            ).show()
            return
        }

        // Hide keyboard
        hideKeyboard()

        // Show loading
        showSearchLoading(true)

        lifecycleScope.launch {
            try {
                // Call both APIs in parallel
                val prediction = RetrofitClient.apiService.getNAVPrediction(fundId)
                val classification = RetrofitClient.apiService.classifyPerformance(fundId)

                // Display results
                displaySearchResults(prediction, classification)

                showSearchLoading(false)

                Snackbar.make(
                    binding.root,
                    "✅ Fund data loaded successfully!",
                    Snackbar.LENGTH_SHORT
                ).show()

            } catch (e: Exception) {
                showSearchLoading(false)

                binding.searchResultCard.visibility = View.GONE

                Snackbar.make(
                    binding.root,
                    "❌ Fund not found or API error: ${e.message}",
                    Snackbar.LENGTH_LONG
                ).setAction("Retry") { performSearch() }
                    .show()
            }
        }
    }

    /**
     * 📊 Display Search Results
     */
    private fun displaySearchResults(
        prediction: NAVPrediction,
        classification: PerformanceClassification
    ) {
        binding.apply {
            searchResultCard.visibility = View.VISIBLE

            // Fund Info
            tvSearchFundName.text = prediction.fundName
            tvSearchFundId.text = prediction.fundId

            // NAV Prediction Data
            tvSearchCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
            tvSearchPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
            tvSearchChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
            tvSearchConfidence.text = String.format("%.1f%%", prediction.confidence)

            // Color code based on change
            val changeColor = if (prediction.changePercentage > 0) {
                getColor(R.color.success_green)
            } else {
                getColor(R.color.error_red)
            }
            tvSearchPredictedNav.setTextColor(changeColor)
            tvSearchChangePercentage.setTextColor(changeColor)

            // Performance Classification
            val (classIcon, classColor, className) = when (classification.performanceClass.uppercase()) {
                "HIGH" -> Triple("⭐", R.color.success_green, "HIGH PERFORMER")
                "MEDIUM" -> Triple("📊", R.color.warning_yellow, "MEDIUM PERFORMER")
                "LOW" -> Triple("📉", R.color.error_red, "LOW PERFORMER")
                else -> Triple("❓", R.color.gray_600, classification.performanceClass)
            }

            tvSearchClassificationIcon.text = classIcon
            tvSearchClassificationIcon.setBackgroundResource(
                when (classification.performanceClass.uppercase()) {
                    "HIGH" -> R.drawable.bg_circle_green
                    "MEDIUM" -> R.drawable.bg_circle_yellow
                    "LOW" -> R.drawable.bg_circle_red
                    else -> R.drawable.bg_circle_gray
                }
            )

            tvSearchClassification.text = className
            tvSearchClassification.setTextColor(getColor(classColor))

            tvSearchProbability.text = "Probability: ${String.format("%.1f%%", classification.confidence * 100)}"

            // Recommendation based on classification
            val recommendation = when (classification.performanceClass.uppercase()) {
                "HIGH" -> "💡 This fund shows strong performance indicators. " +
                        "Predicted to ${if (prediction.changePercentage > 0) "increase" else "decrease"} " +
                        "by ${String.format("%.2f%%", Math.abs(prediction.changePercentage))}. " +
                        "Consider as a good investment opportunity."
                "MEDIUM" -> "💡 This fund has moderate performance. " +
                        "Predicted to ${if (prediction.changePercentage > 0) "increase" else "decrease"} " +
                        "by ${String.format("%.2f%%", Math.abs(prediction.changePercentage))}. " +
                        "Monitor closely before making investment decisions."
                "LOW" -> "⚠️ This fund shows lower performance indicators. " +
                        "Predicted to ${if (prediction.changePercentage > 0) "increase" else "decrease"} " +
                        "by ${String.format("%.2f%%", Math.abs(prediction.changePercentage))}. " +
                        "Exercise caution with this investment."
                else -> "💡 ${classification.recommendation}"
            }

            tvSearchRecommendation.text = recommendation

            // Scroll to results
            binding.scrollView.post {
                binding.scrollView.smoothScrollTo(0, binding.searchResultCard.top)
            }
        }
    }

    /**
     * Hide Keyboard
     */
    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(binding.searchEditText.windowToken, 0)
    }

    /**
     * Show/Hide Search Loading
     */
    private fun showSearchLoading(loading: Boolean) {
        binding.apply {
            btnSearch.isEnabled = !loading
            btnSearch.text = if (loading) "Searching..." else "Search"

            if (loading) {
                searchResultCard.visibility = View.GONE
            }
        }
    }

    /**
     * 🤖 Load AI Insights from Backend
     */
    private fun loadAIInsights() {
        showLoading(true)

        lifecycleScope.launch {
            try {
                // Fetch AI data from backend
                val stats = RetrofitClient.apiService.getPerformanceStats()
                val recommendations = RetrofitClient.apiService.getAIRecommendations()

                // Update UI
                updateOverviewStats(stats)
                updateRecommendations(recommendations)

                showLoading(false)

            } catch (e: Exception) {
                showError("Failed to load AI insights: ${e.message}")
                showLoading(false)
            }
        }
    }

    /**
     * 📊 Show Overview Tab
     */
    private fun showOverview() {
        binding.apply {
            layoutOverview.visibility = View.VISIBLE
            layoutPredictions.visibility = View.GONE
            layoutRecommendations.visibility = View.GONE
            layoutRiskAnalysis.visibility = View.GONE
        }
    }

    /**
     * 🎯 Show Predictions Tab
     */
    private fun showPredictions() {
        binding.apply {
            layoutOverview.visibility = View.GONE
            layoutPredictions.visibility = View.VISIBLE
            layoutRecommendations.visibility = View.GONE
            layoutRiskAnalysis.visibility = View.GONE
        }

        loadNAVPredictions()
    }

    /**
     * 💡 Show Recommendations Tab
     */
    private fun showRecommendations() {
        binding.apply {
            layoutOverview.visibility = View.GONE
            layoutPredictions.visibility = View.GONE
            layoutRecommendations.visibility = View.VISIBLE
            layoutRiskAnalysis.visibility = View.GONE
        }
    }

    /**
     * ⚠️ Show Risk Analysis Tab
     */
    private fun showRiskAnalysis() {
        binding.apply {
            layoutOverview.visibility = View.GONE
            layoutPredictions.visibility = View.GONE
            layoutRecommendations.visibility = View.GONE
            layoutRiskAnalysis.visibility = View.VISIBLE
        }

        loadRiskAnalysis()
    }

    /**
     * Update Overview Statistics with Charts
     */
    private fun updateOverviewStats(stats: PerformanceStats) {
        binding.apply {
            // Model Performance
            tvModelAccuracy.text = String.format("%.1f%%", stats.modelAccuracy)
            tvTotalFunds.text = stats.totalFunds.toString()
            tvHighPerformers.text = stats.highPerformers.toString()

            // Performance Distribution Text
            tvHighCount.text = stats.highPerformers.toString()
            tvMediumCount.text = stats.mediumPerformers.toString()
            tvLowCount.text = stats.lowPerformers.toString()

            // Risk Distribution Text
            tvHighRisk.text = "${stats.highRiskFunds} funds"
            tvMediumRisk.text = "${stats.mediumRiskFunds} funds"
            tvLowRisk.text = "${stats.lowRiskFunds} funds"

            // Update Progress Bars
            progressHighRisk.progress = calculatePercentage(stats.highRiskFunds, stats.totalFunds)
            progressMediumRisk.progress = calculatePercentage(stats.mediumRiskFunds, stats.totalFunds)
            progressLowRisk.progress = calculatePercentage(stats.lowRiskFunds, stats.totalFunds)

            // 📊 Setup Performance Distribution Pie Chart
            setupPerformancePieChart(pieChartPerformance, stats)

            // 📊 Setup Risk Distribution Bar Chart
            setupRiskBarChart(barChartRisk, stats)
        }
    }

    /**
     * 📊 Setup Performance Distribution Pie Chart
     */
    private fun setupPerformancePieChart(chart: PieChart, stats: PerformanceStats) {
        val entries = ArrayList<PieEntry>()
        entries.add(PieEntry(stats.highPerformers.toFloat(), "High"))
        entries.add(PieEntry(stats.mediumPerformers.toFloat(), "Medium"))
        entries.add(PieEntry(stats.lowPerformers.toFloat(), "Low"))

        val dataSet = PieDataSet(entries, "Performance Distribution")
        dataSet.colors = listOf(
            Color.rgb(76, 175, 80),   // Green - High
            Color.rgb(255, 193, 7),    // Yellow - Medium
            Color.rgb(244, 67, 54)     // Red - Low
        )
        dataSet.valueTextSize = 14f
        dataSet.valueTextColor = Color.WHITE
        dataSet.sliceSpace = 3f

        val data = PieData(dataSet)
        data.setValueFormatter(PercentFormatter(chart))

        chart.apply {
            this.data = data
            description.isEnabled = false
            isRotationEnabled = true
            setUsePercentValues(true)
            setEntryLabelColor(Color.BLACK)
            setEntryLabelTextSize(12f)
            centerText = "Performance\nDistribution"
            setCenterTextSize(14f)
            legend.apply {
                isEnabled = true
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
                textSize = 11f
            }
            animateY(1000, Easing.EaseInOutQuad)
            invalidate()
        }
    }

    /**
     * 📊 Setup Risk Distribution Bar Chart
     */
    private fun setupRiskBarChart(chart: BarChart, stats: PerformanceStats) {
        val entries = ArrayList<BarEntry>()
        entries.add(BarEntry(0f, stats.highRiskFunds.toFloat()))
        entries.add(BarEntry(1f, stats.mediumRiskFunds.toFloat()))
        entries.add(BarEntry(2f, stats.lowRiskFunds.toFloat()))

        val dataSet = BarDataSet(entries, "Number of Funds")
        dataSet.colors = listOf(
            Color.rgb(244, 67, 54),   // Red - High Risk
            Color.rgb(255, 193, 7),    // Yellow - Medium Risk
            Color.rgb(76, 175, 80)     // Green - Low Risk
        )
        dataSet.valueTextSize = 12f
        dataSet.valueTextColor = Color.WHITE

        val data = BarData(dataSet)
        data.barWidth = 0.6f

        val labels = listOf("High Risk", "Medium Risk", "Low Risk")

        chart.apply {
            this.data = data
            description.isEnabled = false
            setFitBars(true)

            xAxis.apply {
                valueFormatter = IndexAxisValueFormatter(labels)
                position = XAxis.XAxisPosition.BOTTOM
                granularity = 1f
                textSize = 10f
                setDrawGridLines(false)
            }

            axisLeft.apply {
                textSize = 10f
                axisMinimum = 0f
            }

            axisRight.isEnabled = false

            legend.apply {
                isEnabled = true
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                horizontalAlignment = Legend.LegendHorizontalAlignment.RIGHT
                textSize = 11f
            }

            animateY(1000, Easing.EaseInOutQuad)
            invalidate()
        }
    }

    /**
     * 🎯 Load NAV Predictions
     */
    private fun loadNAVPredictions() {
        lifecycleScope.launch {
            try {
                val predictions = RetrofitClient.apiService.getAllNAVPredictions()

                // Display top 3 predictions
                if (predictions.isNotEmpty()) {
                    updatePredictionCard1(predictions[0])
                }
                if (predictions.size > 1) {
                    updatePredictionCard2(predictions[1])
                }
                if (predictions.size > 2) {
                    updatePredictionCard3(predictions[2])
                }

                // Setup predictions chart
                setupPredictionsChart(binding.barChartPredictions, predictions.take(10))

            } catch (e: Exception) {
                showError("Failed to load predictions: ${e.message}")
            }
        }
    }

    /**
     * 📊 Setup Predictions Bar Chart
     */
    private fun setupPredictionsChart(chart: BarChart, predictions: List<NAVPrediction>) {
        val currentEntries = ArrayList<BarEntry>()
        val predictedEntries = ArrayList<BarEntry>()
        val labels = ArrayList<String>()

        predictions.forEachIndexed { index, prediction ->
            currentEntries.add(BarEntry(index.toFloat(), prediction.currentNAV.toFloat()))
            predictedEntries.add(BarEntry(index.toFloat(), prediction.predictedNAV.toFloat()))
            labels.add(prediction.fundId)
        }

        val currentDataSet = BarDataSet(currentEntries, "Current NAV")
        currentDataSet.color = Color.rgb(66, 165, 245)
        currentDataSet.valueTextSize = 10f
        currentDataSet.valueTextColor = Color.WHITE

        val predictedDataSet = BarDataSet(predictedEntries, "Predicted NAV")
        predictedDataSet.color = Color.rgb(102, 187, 106)
        predictedDataSet.valueTextSize = 10f
        predictedDataSet.valueTextColor = Color.WHITE

        val data = BarData(currentDataSet, predictedDataSet)
        data.barWidth = 0.35f

        chart.apply {
            this.data = data
            description.isEnabled = false
            setFitBars(true)
            groupBars(-0.5f, 0.3f, 0.03f)

            xAxis.apply {
                valueFormatter = IndexAxisValueFormatter(labels)
                position = XAxis.XAxisPosition.BOTTOM
                granularity = 1f
                textSize = 10f
                axisMinimum = 0f
            }

            axisLeft.apply {
                textSize = 10f
                axisMinimum = 0f
            }

            axisRight.isEnabled = false

            legend.apply {
                isEnabled = true
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                horizontalAlignment = Legend.LegendHorizontalAlignment.RIGHT
                textSize = 11f
            }

            animateY(1000, Easing.EaseInOutQuad)
            invalidate()
        }
    }

    /**
     * Update Recommendations
     */
    private fun updateRecommendations(recommendations: List<AIRecommendation>) {
        recommendationsAdapter = RecommendedFundsAdapter(recommendations) { fund ->
            Snackbar.make(
                binding.root,
                "Opening ${fund.fundName}...",
                Snackbar.LENGTH_SHORT
            ).show()
        }
        binding.recyclerViewRecommendations.adapter = recommendationsAdapter
    }

    /**
     * Update Prediction Card 1
     */
    private fun updatePredictionCard1(prediction: NAVPrediction) {
        binding.cardPrediction1.root.visibility = View.VISIBLE
        binding.cardPrediction1.apply {
            tvFundName.text = prediction.fundName
            tvFundId.text = prediction.fundId
            tvCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
            tvPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
            tvChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
            tvConfidence.text = String.format("%.1f%%", prediction.confidence)
            tvPredictionDate.text = formatDate(prediction.predictionDate)

            tvTrendIcon.text = if (prediction.changePercentage > 0) "📈" else "📉"

            val changeColor = if (prediction.changePercentage > 0) {
                getColor(R.color.success_green)
            } else {
                getColor(R.color.error_red)
            }
            tvChangePercentage.setTextColor(changeColor)
            tvPredictedNav.setTextColor(changeColor)
        }
    }

    /**
     * Update Prediction Card 2
     */
    private fun updatePredictionCard2(prediction: NAVPrediction) {
        binding.cardPrediction2.root.visibility = View.VISIBLE
        binding.cardPrediction2.apply {
            tvFundName.text = prediction.fundName
            tvFundId.text = prediction.fundId
            tvCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
            tvPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
            tvChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
            tvConfidence.text = String.format("%.1f%%", prediction.confidence)
            tvPredictionDate.text = formatDate(prediction.predictionDate)

            tvTrendIcon.text = if (prediction.changePercentage > 0) "📈" else "📉"

            val changeColor = if (prediction.changePercentage > 0) {
                getColor(R.color.success_green)
            } else {
                getColor(R.color.error_red)
            }
            tvChangePercentage.setTextColor(changeColor)
            tvPredictedNav.setTextColor(changeColor)
        }
    }

    /**
     * Update Prediction Card 3
     */
    private fun updatePredictionCard3(prediction: NAVPrediction) {
        binding.cardPrediction3.root.visibility = View.VISIBLE
        binding.cardPrediction3.apply {
            tvFundName.text = prediction.fundName
            tvFundId.text = prediction.fundId
            tvCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
            tvPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
            tvChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
            tvConfidence.text = String.format("%.1f%%", prediction.confidence)
            tvPredictionDate.text = formatDate(prediction.predictionDate)

            tvTrendIcon.text = if (prediction.changePercentage > 0) "📈" else "📉"

            val changeColor = if (prediction.changePercentage > 0) {
                getColor(R.color.success_green)
            } else {
                getColor(R.color.error_red)
            }
            tvChangePercentage.setTextColor(changeColor)
            tvPredictedNav.setTextColor(changeColor)
        }
    }

    /**
     * Load Risk Analysis with Chart
     */
    private fun loadRiskAnalysis() {
        lifecycleScope.launch {
            try {
                val riskData = RetrofitClient.apiService.getRiskAnalysis()

                binding.apply {
                    // Update risk warnings
                    tvHighRiskWarning.text = "⚠️ ${riskData.highRiskCount} funds flagged as high risk"
                    tvAvgExpenseRatio.text = String.format("%.2f%%", riskData.avgHighRiskExpense)
                    tvAvgPerfFee.text = String.format("%.2f%%", riskData.avgHighRiskPerfFee)
                    tvRiskRecommendation.text = riskData.recommendation

                    // 📊 Setup Risk Metrics Horizontal Bar Chart
                    setupRiskMetricsChart(horizontalBarChartRisk, riskData)
                }

            } catch (e: Exception) {
                showError("Failed to load risk analysis: ${e.message}")
            }
        }
    }

    /**
     * 📊 Setup Risk Metrics Horizontal Bar Chart
     */
    private fun setupRiskMetricsChart(chart: BarChart, riskData: RiskAssessment) {
        val entries = ArrayList<BarEntry>()
        entries.add(BarEntry(0f, riskData.avgHighRiskExpense.toFloat()))
        entries.add(BarEntry(1f, riskData.avgHighRiskPerfFee.toFloat()))

        val dataSet = BarDataSet(entries, "Risk Metrics (%)")
        dataSet.colors = listOf(
            Color.rgb(244, 67, 54),  // Red
            Color.rgb(255, 152, 0)   // Orange
        )
        dataSet.valueTextSize = 12f
        dataSet.valueTextColor = Color.WHITE

        val data = BarData(dataSet)
        data.barWidth = 0.7f

        val labels = listOf("Avg Expense\nRatio", "Avg Performance\nFee")

        chart.apply {
            this.data = data
            description.isEnabled = false
            setFitBars(true)

            xAxis.apply {
                valueFormatter = IndexAxisValueFormatter(labels)
                position = XAxis.XAxisPosition.BOTTOM
                granularity = 1f
                textSize = 11f
                setDrawGridLines(false)
            }

            axisLeft.apply {
                textSize = 11f
                axisMinimum = 0f
                axisMaximum = 100f
            }

            axisRight.isEnabled = false
            legend.isEnabled = false

            animateY(1000, Easing.EaseInOutQuad)
            invalidate()
        }
    }

    /**
     * Format Date String
     */
    private fun formatDate(dateString: String): String {
        return try {
            val parts = dateString.split("-")
            val year = parts[0]
            val month = when (parts[1]) {
                "01" -> "Jan"; "02" -> "Feb"; "03" -> "Mar"; "04" -> "Apr"
                "05" -> "May"; "06" -> "Jun"; "07" -> "Jul"; "08" -> "Aug"
                "09" -> "Sep"; "10" -> "Oct"; "11" -> "Nov"; "12" -> "Dec"
                else -> parts[1]
            }
            "$month $year"
        } catch (e: Exception) {
            dateString
        }
    }

    /**
     * Calculate Percentage
     */
    private fun calculatePercentage(part: Int, total: Int): Int {
        return if (total == 0) 0 else ((part.toDouble() / total) * 100).toInt()
    }

    /**
     * Show/Hide Loading
     */
    private fun showLoading(loading: Boolean) {
        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        binding.scrollView.visibility = if (loading) View.GONE else View.VISIBLE
    }

    /**
     * Show Error
     */
    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Retry") { loadAIInsights() }
            .show()
    }

    override fun onBackPressed() {
        super.onBackPressed()
        overridePendingTransition(
            android.R.anim.slide_in_left,
            android.R.anim.slide_out_right
        )
    }
}
//package com.example.fundbank
//
//import adapters.RecommendedFundsAdapter
//import android.graphics.Color
//import android.os.Bundle
//import androidx.appcompat.app.AppCompatActivity
//import android.view.View
//import androidx.lifecycle.lifecycleScope
//import androidx.recyclerview.widget.LinearLayoutManager
//import api.RetrofitClient
//import com.example.fundbank.databinding.ActivityAllnsightsBinding
//import com.github.mikephil.charting.animation.Easing
//import com.github.mikephil.charting.charts.BarChart
//import com.github.mikephil.charting.charts.PieChart
//import com.github.mikephil.charting.components.Legend
//import com.github.mikephil.charting.components.XAxis
//import com.github.mikephil.charting.data.*
//import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
//import com.github.mikephil.charting.formatter.PercentFormatter
//import com.google.android.material.snackbar.Snackbar
//import com.google.android.material.tabs.TabLayout
//import kotlinx.coroutines.launch
//import models.AIRecommendation
//import models.PerformanceStats
//import models.NAVPrediction
//import models.RiskAssessment
//
//class AIInsightsActivity : AppCompatActivity() {
//
//    private lateinit var binding: ActivityAllnsightsBinding
//    private lateinit var recommendationsAdapter: RecommendedFundsAdapter
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        binding = ActivityAllnsightsBinding.inflate(layoutInflater)
//        setContentView(binding.root)
//
//        setupToolbar()
//        setupTabs()
//        setupRecyclerView()
//        loadAIInsights()
//    }
//
//    private fun setupToolbar() {
//        setSupportActionBar(binding.toolbar)
//        supportActionBar?.apply {
//            setDisplayHomeAsUpEnabled(true)
//            setDisplayShowHomeEnabled(true)
//            title = "AI Insights"
//        }
//
//        binding.toolbar.setNavigationOnClickListener {
//            onBackPressed()
//        }
//    }
//
//    private fun setupTabs() {
//        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("📊 Overview"))
//        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("🎯 Predictions"))
//        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("💡 Recommendations"))
//        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("⚠️ Risk Analysis"))
//
//        binding.tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
//            override fun onTabSelected(tab: TabLayout.Tab?) {
//                when (tab?.position) {
//                    0 -> showOverview()
//                    1 -> showPredictions()
//                    2 -> showRecommendations()
//                    3 -> showRiskAnalysis()
//                }
//            }
//            override fun onTabUnselected(tab: TabLayout.Tab?) {}
//            override fun onTabReselected(tab: TabLayout.Tab?) {}
//        })
//    }
//
//    private fun setupRecyclerView() {
//        recommendationsAdapter = RecommendedFundsAdapter(emptyList()) { fund ->
//            // Handle fund click
//            Snackbar.make(
//                binding.root,
//                "Opening ${fund.fundName}...",
//                Snackbar.LENGTH_SHORT
//            ).show()
//        }
//
//        binding.recyclerViewRecommendations.apply {
//            layoutManager = LinearLayoutManager(this@AIInsightsActivity)
//            adapter = recommendationsAdapter
//        }
//    }
//
//    /**
//     * 🤖 Load AI Insights from Backend
//     */
//    private fun loadAIInsights() {
//        showLoading(true)
//
//        lifecycleScope.launch {
//            try {
//                // Fetch AI data from backend
//                val stats = RetrofitClient.apiService.getPerformanceStats()
//                val recommendations = RetrofitClient.apiService.getAIRecommendations()
//
//                // Update UI
//                updateOverviewStats(stats)
//                updateRecommendations(recommendations)
//
//                showLoading(false)
//
//            } catch (e: Exception) {
//                showError("Failed to load AI insights: ${e.message}")
//                showLoading(false)
//            }
//        }
//    }
//
//    /**
//     * 📊 Show Overview Tab
//     */
//    private fun showOverview() {
//        binding.apply {
//            layoutOverview.visibility = View.VISIBLE
//            layoutPredictions.visibility = View.GONE
//            layoutRecommendations.visibility = View.GONE
//            layoutRiskAnalysis.visibility = View.GONE
//        }
//    }
//
//    /**
//     * 🎯 Show Predictions Tab
//     */
//    private fun showPredictions() {
//        binding.apply {
//            layoutOverview.visibility = View.GONE
//            layoutPredictions.visibility = View.VISIBLE
//            layoutRecommendations.visibility = View.GONE
//            layoutRiskAnalysis.visibility = View.GONE
//        }
//
//        loadNAVPredictions()
//    }
//
//    /**
//     * 💡 Show Recommendations Tab
//     */
//    private fun showRecommendations() {
//        binding.apply {
//            layoutOverview.visibility = View.GONE
//            layoutPredictions.visibility = View.GONE
//            layoutRecommendations.visibility = View.VISIBLE
//            layoutRiskAnalysis.visibility = View.GONE
//        }
//    }
//
//    /**
//     * ⚠️ Show Risk Analysis Tab
//     */
//    private fun showRiskAnalysis() {
//        binding.apply {
//            layoutOverview.visibility = View.GONE
//            layoutPredictions.visibility = View.GONE
//            layoutRecommendations.visibility = View.GONE
//            layoutRiskAnalysis.visibility = View.VISIBLE
//        }
//
//        loadRiskAnalysis()
//    }
//
//    /**
//     * Update Overview Statistics with Charts
//     */
//    private fun updateOverviewStats(stats: PerformanceStats) {
//        binding.apply {
//            // Model Performance
//            tvModelAccuracy.text = String.format("%.1f%%", stats.modelAccuracy)
//            tvTotalFunds.text = stats.totalFunds.toString()
//            tvHighPerformers.text = stats.highPerformers.toString()
//
//            // Performance Distribution Text
//            tvHighCount.text = stats.highPerformers.toString()
//            tvMediumCount.text = stats.mediumPerformers.toString()
//            tvLowCount.text = stats.lowPerformers.toString()
//
//            // Risk Distribution Text
//            tvHighRisk.text = "${stats.highRiskFunds} funds"
//            tvMediumRisk.text = "${stats.mediumRiskFunds} funds"
//            tvLowRisk.text = "${stats.lowRiskFunds} funds"
//
//            // Update Progress Bars
//            progressHighRisk.progress = calculatePercentage(stats.highRiskFunds, stats.totalFunds)
//            progressMediumRisk.progress = calculatePercentage(stats.mediumRiskFunds, stats.totalFunds)
//            progressLowRisk.progress = calculatePercentage(stats.lowRiskFunds, stats.totalFunds)
//
//            // 📊 Setup Performance Distribution Pie Chart
//            setupPerformancePieChart(pieChartPerformance, stats)
//
//            // 📊 Setup Risk Distribution Bar Chart
//            setupRiskBarChart(barChartRisk, stats)
//        }
//    }
//
//    /**
//     * 📊 Setup Performance Distribution Pie Chart
//     */
//    private fun setupPerformancePieChart(chart: PieChart, stats: PerformanceStats) {
//        val entries = ArrayList<PieEntry>()
//        entries.add(PieEntry(stats.highPerformers.toFloat(), "High"))
//        entries.add(PieEntry(stats.mediumPerformers.toFloat(), "Medium"))
//        entries.add(PieEntry(stats.lowPerformers.toFloat(), "Low"))
//
//        val dataSet = PieDataSet(entries, "Performance Distribution")
//        dataSet.colors = listOf(
//            Color.rgb(76, 175, 80),   // Green - High
//            Color.rgb(255, 193, 7),    // Yellow - Medium
//            Color.rgb(244, 67, 54)     // Red - Low
//        )
//        dataSet.valueTextSize = 14f
//        dataSet.valueTextColor = Color.WHITE
//        dataSet.sliceSpace = 3f
//
//        val data = PieData(dataSet)
//        data.setValueFormatter(PercentFormatter(chart))
//
//        chart.apply {
//            this.data = data
//            description.isEnabled = false
//            isDrawHoleEnabled = true
//            setHoleColor(Color.TRANSPARENT)
//            holeRadius = 40f
//            transparentCircleRadius = 45f
//            setDrawCenterText(true)
//            centerText = "Performance\nDistribution"
//            setCenterTextSize(12f)
//            legend.isEnabled = true
//            legend.verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
//            legend.horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
//            legend.textSize = 12f
//            animateY(1000, Easing.EaseInOutQuad)
//            invalidate()
//        }
//    }
//
//    /**
//     * 📊 Setup Risk Distribution Bar Chart
//     */
//    private fun setupRiskBarChart(chart: BarChart, stats: PerformanceStats) {
//        val entries = ArrayList<BarEntry>()
//        entries.add(BarEntry(0f, stats.highRiskFunds.toFloat()))
//        entries.add(BarEntry(1f, stats.mediumRiskFunds.toFloat()))
//        entries.add(BarEntry(2f, stats.lowRiskFunds.toFloat()))
//
//        val dataSet = BarDataSet(entries, "Risk Distribution")
//        dataSet.colors = listOf(
//            Color.rgb(244, 67, 54),    // Red - High Risk
//            Color.rgb(255, 193, 7),     // Yellow - Medium Risk
//            Color.rgb(76, 175, 80)      // Green - Low Risk
//        )
//        dataSet.valueTextSize = 12f
//        dataSet.valueTextColor = Color.BLACK
//
//        val data = BarData(dataSet)
//        data.barWidth = 0.6f
//
//        val labels = listOf("High Risk", "Medium Risk", "Low Risk")
//
//        chart.apply {
//            this.data = data
//            description.isEnabled = false
//            setFitBars(true)
//
//            xAxis.apply {
//                valueFormatter = IndexAxisValueFormatter(labels)
//                position = XAxis.XAxisPosition.BOTTOM
//                granularity = 1f
//                textSize = 11f
//                setDrawGridLines(false)
//            }
//
//            axisLeft.apply {
//                textSize = 11f
//                axisMinimum = 0f
//                setDrawGridLines(true)
//            }
//
//            axisRight.isEnabled = false
//
//            legend.isEnabled = false
//
//            animateY(1000, Easing.EaseInOutQuad)
//            invalidate()
//        }
//    }
//
//    /**
//     * Update Recommendations List
//     */
//    private fun updateRecommendations(recommendations: List<AIRecommendation>) {
//        recommendationsAdapter.updateData(recommendations)
//        binding.tvRecommendationCount.text = "${recommendations.size} AI-Recommended Funds"
//    }
//
//    /**
//     * Load NAV Predictions with Chart
//     */
//    private fun loadNAVPredictions() {
//        lifecycleScope.launch {
//            try {
//                val predictions = RetrofitClient.apiService.getAllNAVPredictions()
//
//                // Display top 3 prediction cards
//                if (predictions.isNotEmpty()) {
//                    predictions.take(3).forEachIndexed { index, pred ->
//                        when (index) {
//                            0 -> updatePredictionCard1(pred)
//                            1 -> updatePredictionCard2(pred)
//                            2 -> updatePredictionCard3(pred)
//                        }
//                    }
//                }
//
//                // 📊 Setup Predictions Bar Chart (Top 5)
//                setupPredictionsBarChart(binding.barChartPredictions, predictions.take(5))
//
//            } catch (e: Exception) {
//                showError("Failed to load predictions: ${e.message}")
//            }
//        }
//    }
//
//    /**
//     * 📊 Setup Predictions Bar Chart
//     */
//    private fun setupPredictionsBarChart(chart: BarChart, predictions: List<NAVPrediction>) {
//        if (predictions.isEmpty()) {
//            chart.visibility = View.GONE
//            return
//        }
//
//        chart.visibility = View.VISIBLE
//
//        val currentEntries = ArrayList<BarEntry>()
//        val predictedEntries = ArrayList<BarEntry>()
//        val labels = ArrayList<String>()
//
//        predictions.forEachIndexed { index, pred ->
//            currentEntries.add(BarEntry(index.toFloat(), pred.currentNAV.toFloat()))
//            predictedEntries.add(BarEntry(index.toFloat(), pred.predictedNAV.toFloat()))
//            labels.add(pred.fundId)
//        }
//
//        val currentDataSet = BarDataSet(currentEntries, "Current NAV")
//        currentDataSet.color = Color.rgb(33, 150, 243) // Blue
//        currentDataSet.valueTextSize = 10f
//
//        val predictedDataSet = BarDataSet(predictedEntries, "Predicted NAV")
//        predictedDataSet.color = Color.rgb(76, 175, 80) // Green
//        predictedDataSet.valueTextSize = 10f
//
//        val data = BarData(currentDataSet, predictedDataSet)
//        data.barWidth = 0.35f
//
//        chart.apply {
//            this.data = data
//            description.isEnabled = false
//            setFitBars(true)
//            groupBars(0f, 0.3f, 0.03f)
//
//            xAxis.apply {
//                valueFormatter = IndexAxisValueFormatter(labels)
//                position = XAxis.XAxisPosition.BOTTOM
//                granularity = 1f
//                textSize = 10f
//                setDrawGridLines(false)
//                labelRotationAngle = -45f
//            }
//
//            axisLeft.apply {
//                textSize = 10f
//                axisMinimum = 0f
//            }
//
//            axisRight.isEnabled = false
//
//            legend.apply {
//                isEnabled = true
//                verticalAlignment = Legend.LegendVerticalAlignment.TOP
//                horizontalAlignment = Legend.LegendHorizontalAlignment.RIGHT
//                textSize = 11f
//            }
//
//            animateY(1000, Easing.EaseInOutQuad)
//            invalidate()
//        }
//    }
//
//    /**
//     * Update Prediction Card 1
//     */
//    private fun updatePredictionCard1(prediction: NAVPrediction) {
//        binding.cardPrediction1.root.visibility = View.VISIBLE
//        binding.cardPrediction1.apply {
//            tvFundName.text = prediction.fundName
//            tvFundId.text = prediction.fundId
//            tvCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
//            tvPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
//            tvChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
//            tvConfidence.text = String.format("%.1f%%", prediction.confidence)
//            tvPredictionDate.text = formatDate(prediction.predictionDate)
//
//            tvTrendIcon.text = if (prediction.changePercentage > 0) "📈" else "📉"
//
//            val changeColor = if (prediction.changePercentage > 0) {
//                getColor(R.color.success_green)
//            } else {
//                getColor(R.color.error_red)
//            }
//            tvChangePercentage.setTextColor(changeColor)
//            tvPredictedNav.setTextColor(changeColor)
//        }
//    }
//
//    /**
//     * Update Prediction Card 2
//     */
//    private fun updatePredictionCard2(prediction: NAVPrediction) {
//        binding.cardPrediction2.root.visibility = View.VISIBLE
//        binding.cardPrediction2.apply {
//            tvFundName.text = prediction.fundName
//            tvFundId.text = prediction.fundId
//            tvCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
//            tvPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
//            tvChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
//            tvConfidence.text = String.format("%.1f%%", prediction.confidence)
//            tvPredictionDate.text = formatDate(prediction.predictionDate)
//
//            tvTrendIcon.text = if (prediction.changePercentage > 0) "📈" else "📉"
//
//            val changeColor = if (prediction.changePercentage > 0) {
//                getColor(R.color.success_green)
//            } else {
//                getColor(R.color.error_red)
//            }
//            tvChangePercentage.setTextColor(changeColor)
//            tvPredictedNav.setTextColor(changeColor)
//        }
//    }
//
//    /**
//     * Update Prediction Card 3
//     */
//    private fun updatePredictionCard3(prediction: NAVPrediction) {
//        binding.cardPrediction3.root.visibility = View.VISIBLE
//        binding.cardPrediction3.apply {
//            tvFundName.text = prediction.fundName
//            tvFundId.text = prediction.fundId
//            tvCurrentNav.text = "$${String.format("%.2f", prediction.currentNAV)}"
//            tvPredictedNav.text = "$${String.format("%.2f", prediction.predictedNAV)}"
//            tvChangePercentage.text = String.format("%+.2f%%", prediction.changePercentage)
//            tvConfidence.text = String.format("%.1f%%", prediction.confidence)
//            tvPredictionDate.text = formatDate(prediction.predictionDate)
//
//            tvTrendIcon.text = if (prediction.changePercentage > 0) "📈" else "📉"
//
//            val changeColor = if (prediction.changePercentage > 0) {
//                getColor(R.color.success_green)
//            } else {
//                getColor(R.color.error_red)
//            }
//            tvChangePercentage.setTextColor(changeColor)
//            tvPredictedNav.setTextColor(changeColor)
//        }
//    }
//
//    /**
//     * Load Risk Analysis with Chart
//     */
//    private fun loadRiskAnalysis() {
//        lifecycleScope.launch {
//            try {
//                val riskData = RetrofitClient.apiService.getRiskAnalysis()
//
//                binding.apply {
//                    // Update risk warnings
//                    tvHighRiskWarning.text = "⚠️ ${riskData.highRiskCount} funds flagged as high risk"
//                    tvAvgExpenseRatio.text = String.format("%.2f%%", riskData.avgHighRiskExpense)
//                    tvAvgPerfFee.text = String.format("%.2f%%", riskData.avgHighRiskPerfFee)
//                    tvRiskRecommendation.text = riskData.recommendation
//
//                    // 📊 Setup Risk Metrics Horizontal Bar Chart
//                    setupRiskMetricsChart(horizontalBarChartRisk, riskData)
//                }
//
//            } catch (e: Exception) {
//                showError("Failed to load risk analysis: ${e.message}")
//            }
//        }
//    }
//
//    /**
//     * 📊 Setup Risk Metrics Horizontal Bar Chart
//     */
//    private fun setupRiskMetricsChart(chart: BarChart, riskData: RiskAssessment) {
//        val entries = ArrayList<BarEntry>()
//        entries.add(BarEntry(0f, riskData.avgHighRiskExpense.toFloat()))
//        entries.add(BarEntry(1f, riskData.avgHighRiskPerfFee.toFloat()))
//
//        val dataSet = BarDataSet(entries, "Risk Metrics (%)")
//        dataSet.colors = listOf(
//            Color.rgb(244, 67, 54),  // Red
//            Color.rgb(255, 152, 0)   // Orange
//        )
//        dataSet.valueTextSize = 12f
//        dataSet.valueTextColor = Color.WHITE
//
//        val data = BarData(dataSet)
//        data.barWidth = 0.7f
//
//        val labels = listOf("Avg Expense\nRatio", "Avg Performance\nFee")
//
//        chart.apply {
//            this.data = data
//            description.isEnabled = false
//            setFitBars(true)
//
//            xAxis.apply {
//                valueFormatter = IndexAxisValueFormatter(labels)
//                position = XAxis.XAxisPosition.BOTTOM
//                granularity = 1f
//                textSize = 11f
//                setDrawGridLines(false)
//            }
//
//            axisLeft.apply {
//                textSize = 11f
//                axisMinimum = 0f
//                axisMaximum = 100f
//            }
//
//            axisRight.isEnabled = false
//            legend.isEnabled = false
//
//            animateY(1000, Easing.EaseInOutQuad)
//            invalidate()
//        }
//    }
//
//    /**
//     * Format Date String
//     */
//    private fun formatDate(dateString: String): String {
//        return try {
//            val parts = dateString.split("-")
//            val year = parts[0]
//            val month = when (parts[1]) {
//                "01" -> "Jan"; "02" -> "Feb"; "03" -> "Mar"; "04" -> "Apr"
//                "05" -> "May"; "06" -> "Jun"; "07" -> "Jul"; "08" -> "Aug"
//                "09" -> "Sep"; "10" -> "Oct"; "11" -> "Nov"; "12" -> "Dec"
//                else -> parts[1]
//            }
//            "$month $year"
//        } catch (e: Exception) {
//            dateString
//        }
//    }
//
//    /**
//     * Calculate Percentage
//     */
//    private fun calculatePercentage(part: Int, total: Int): Int {
//        return if (total == 0) 0 else ((part.toDouble() / total) * 100).toInt()
//    }
//
//    /**
//     * Show/Hide Loading
//     */
//    private fun showLoading(loading: Boolean) {
//        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
//        binding.scrollView.visibility = if (loading) View.GONE else View.VISIBLE
//    }
//
//    /**
//     * Show Error
//     */
//    private fun showError(message: String) {
//        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
//            .setAction("Retry") { loadAIInsights() }
//            .show()
//    }
//
//    override fun onBackPressed() {
//        super.onBackPressed()
//        overridePendingTransition(
//            android.R.anim.slide_in_left,
//            android.R.anim.slide_out_right
//        )
//    }
//}
//
