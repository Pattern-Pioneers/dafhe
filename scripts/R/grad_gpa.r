# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(viridis)
library(patchwork)
library(scales)
library(hrbrthemes)
library(ggsci)
library(plotly)

# Set seed for reproducibility
set.seed(123)

# Read the data
grads_data <- read.csv("Z:\\PROJECT\\dafhe\\grads_gpa.csv")

# Data preparation
grads_data <- grads_data %>%
  # Convert term codes to years
  mutate(
    grad_year = as.integer(substr(TERM_CODE_GRAD, 1, 4)),
    admit_year = as.integer(substr(TERM_CODE_ADMIT, 1, 4)),
    
    # Calculate years to graduate
    years_to_graduate = grad_year - admit_year,
    
    # Create better labels for degree codes - CORRECTED MAPPING
    degree_class = case_when(
      CLASS_OF_DEGREE_CODE == "F" ~ "First Class",
      CLASS_OF_DEGREE_CODE == "SU" ~ "Upper Second Class",
      CLASS_OF_DEGREE_CODE == "SL" ~ "Lower Second Class",
      CLASS_OF_DEGREE_CODE == "P" ~ "Pass",
      TRUE ~ "Unknown"
    ),
    
    # Simplify program names for plotting
    program_short = case_when(
      STU_DEGREE_CODE == "BSC Computer Science (Major)" ~ "CS (Major)",
      STU_DEGREE_CODE == "BSC Computer Science (Spec)" ~ "CS (Spec)",
      STU_DEGREE_CODE == "BSC Computer Science and Management" ~ "CS & Mgmt",
      STU_DEGREE_CODE == "BSC Information Technology (Major)" ~ "IT (Major)",
      STU_DEGREE_CODE == "BSC Information Technology (Spec)" ~ "IT (Spec)",
      TRUE ~ STU_DEGREE_CODE
    ),
    
    # Create a field to categorize as CS or IT
    field = ifelse(grepl("Computer Science", STU_DEGREE_CODE), "Computer Science", "Information Technology")
  ) %>%
  # Order factor for plotting
  mutate(
    degree_class = factor(degree_class, 
                         levels = c("First Class", "Upper Second Class", "Lower Second Class", "Pass")),
    program_short = factor(program_short,
                         levels = c("CS (Major)", "CS (Spec)", "CS & Mgmt", "IT (Major)", "IT (Spec)"))
  )

# Set a custom theme for all plots
theme_custom <- theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12, color = "gray30"),
    axis.title = element_text(size = 11, color = "gray30"),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(20, 20, 20, 20)
  )

# Theme for ggplot
theme_set(theme_custom)

# Color palettes - make sure to use GENDER (uppercase) to match the dataset
gender_palette <- c("F" = "#FF6B9F", "M" = "#4169E1")
field_palette <- c("Computer Science" = "#7570B3", "Information Technology" = "#E7298A")
class_palette <- pal_npg("nrc")(4)

# 1. GPA Distribution by Gender
p1 <- ggplot(grads_data, aes(x = GPA, fill = GENDER)) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values = gender_palette) +
  labs(
    title = "GPA Distribution by Gender",
    subtitle = "Density plot showing overall GPA distributions from 2000 to 2024",
    x = "GPA", 
    y = "Density"
  ) +
  scale_x_continuous(limits = c(2, 4.3), breaks = seq(2, 4.3, 0.5)) +
  theme(legend.position = "top")

# 2. Degree Class Distribution by Program
p2 <- grads_data %>%
  ggplot(aes(x = program_short, fill = degree_class)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = class_palette, name = "Degree Class") +
  scale_y_continuous(labels = percent_format()) +
  labs(
    title = "Degree Classifications by Program",
    subtitle = "Proportion of students achieving each degree class",
    x = NULL,
    y = "Percentage"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "right"
  )

# 3. Average GPA Trends Over Time by Field
p3 <- grads_data %>%
  group_by(grad_year, field) %>%
  summarise(avg_gpa = mean(GPA, na.rm = TRUE), .groups = 'drop') %>%
  ggplot(aes(x = grad_year, y = avg_gpa, color = field, group = field)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5) +
  scale_color_manual(values = field_palette, name = "Field") +
  scale_x_continuous(breaks = seq(2000, 2024, 4)) +
  scale_y_continuous(limits = c(2.5, 3.5), breaks = seq(2.5, 3.5, 0.25)) +
  labs(
    title = "Average GPA Trends Over Time",
    subtitle = "Evolution of performance across CS and IT fields",
    x = "Graduation Year",
    y = "Average GPA"
  )

# 4. Gender Representation Across Programs
p4 <- grads_data %>%
  group_by(program_short, GENDER) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(program_short) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  filter(GENDER == "F") %>%
  ggplot(aes(x = reorder(program_short, percentage), y = percentage, fill = program_short)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.1f%%", percentage)), 
            position = position_stack(vjust = 0.5), 
            color = "white", fontface = "bold") +
  scale_fill_viridis_d(option = "D", begin = 0.3, end = 0.8) +
  labs(
    title = "Female Representation Across Programmes",
    subtitle = "Percentage of female students in each programme",
    x = NULL,
    y = "Female Students (%)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "none"
  )

# 5. Years to Graduation by Gender and Program
p5 <- grads_data %>%
  filter(years_to_graduate <= 8) %>% # Filter out potential data errors
  ggplot(aes(x = program_short, y = years_to_graduate, fill = GENDER)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = gender_palette) +
  labs(
    title = "Time to Graduation by Program and Gender",
    subtitle = "Distribution of years taken to complete degrees",
    x = NULL,
    y = "Years to Graduate"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "top"
  )

# 6. Average GPA by Program and Gender
p6 <- grads_data %>%
  group_by(program_short, GENDER) %>%
  summarise(avg_gpa = mean(GPA, na.rm = TRUE), .groups = 'drop') %>%
  ggplot(aes(x = program_short, y = avg_gpa, fill = GENDER)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.2f", avg_gpa)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = gender_palette) +
  labs(
    title = "Average GPA by Program and Gender",
    subtitle = "From 2000 to 2024",
    x = NULL,
    y = "Average GPA"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "top"
  ) +
  coord_cartesian(ylim = c(2.75, 3.5))

# 7. Gender Proportion Over Time (replacing First Class Degrees plot)
p7 <- grads_data %>%
  group_by(grad_year, GENDER) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(grad_year) %>%
  mutate(proportion = count / sum(count) * 100) %>%
  ggplot(aes(x = grad_year, y = proportion, color = GENDER, group = GENDER)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  scale_color_manual(values = gender_palette) +
  scale_x_continuous(breaks = seq(2000, 2024, 4)) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 100)) +
  labs(
    title = "Gender Proportion Trends Over Time",
    subtitle = "Changes in the female to male ratio of graduates from 2000-2024",
    x = "Graduation Year",
    y = "Percentage of Graduates"
  ) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray50", alpha = 0.7) +
  theme(legend.position = "top")

# 8. GPA Heatmap by Admit/Grad Year
p8 <- grads_data %>%
  group_by(admit_year, grad_year) %>%
  summarise(avg_gpa = mean(GPA, na.rm = TRUE),
            count = n(), .groups = 'drop') %>%
  filter(count > 5) %>%
  ggplot(aes(x = grad_year, y = admit_year, fill = avg_gpa)) +
  geom_tile() +
  scale_fill_viridis(option = "C", name = "Avg GPA") +
  scale_x_continuous(breaks = seq(2000, 2024, 4)) +
  scale_y_continuous(breaks = seq(2000, 2024, 4)) +
  labs(
    title = "GPA Heatmap by Admission/Graduation Year",
    subtitle = "Performance patterns across different cohorts",
    x = "Graduation Year",
    y = "Admission Year"
  ) +
  theme(
    legend.position = "right",
    panel.grid = element_blank()
  )

# Add a new plot analyzing degree classifications by graduation year
p9 <- grads_data %>%
  group_by(grad_year, degree_class) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(grad_year) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  ggplot(aes(x = grad_year, y = percentage, fill = degree_class)) +
  geom_area(alpha = 0.8, position = "stack") +
  scale_fill_manual(values = class_palette, name = "Degree Class") +
  scale_x_continuous(breaks = seq(2000, 2024, 4)) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title = "Degree Classifications Over Time",
    subtitle = "Trends in academic performance standards",
    x = "Graduation Year",
    y = "Percentage"
  ) +
  theme(legend.position = "right")

# Combine into a dashboard
# First row
row1 <- p1 + p3 + plot_layout(widths = c(1, 1.5))

# Second row
row2 <- p2 + p4 + plot_layout(widths = c(1.5, 1))

# Third row
row3 <- p5 + p6 + plot_layout(widths = c(1, 1))

# Fourth row
row4 <- p7 + p9 + plot_layout(widths = c(1, 1))

# Fifth row (added the heatmap as its own row)
row5 <- p8 + plot_spacer() + plot_layout(widths = c(2, 0.1))

# Combine all rows
dashboard <- row1 / row2 / row3 / row4 / row5 +
  plot_annotation(
    title = "Student Academic Performance Dashboard",
    subtitle = "Analysis of graduation data from 2000-2024 with LLM-generated distributions",
    theme = theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(0, 0, 20, 0))
    )
  )

# Save the dashboard to a file
ggsave("Z:\\PROJECT\\dafhe\\student_performance_dashboard.png", 
       dashboard, 
       width = 20, 
       height = 28, 
       dpi = 300)

# Print a success message
print("Dashboard has been created and saved successfully!")

# BONUS: Interactive version of GPA by program and gender
interactive_plot <- grads_data %>%
  group_by(program_short, GENDER) %>%
  summarise(
    avg_gpa = mean(GPA, na.rm = TRUE),
    median_gpa = median(GPA, na.rm = TRUE),
    min_gpa = min(GPA, na.rm = TRUE),
    max_gpa = max(GPA, na.rm = TRUE),
    sd_gpa = sd(GPA, na.rm = TRUE),
    count = n(),
    .groups = 'drop'
  ) %>%
  plot_ly(
    x = ~program_short,
    y = ~avg_gpa,
    color = ~GENDER,
    colors = gender_palette,
    type = "bar",
    error_y = list(
      type = "data",
      array = ~sd_gpa/sqrt(count),
      visible = TRUE
    ),
    hoverinfo = "text",
    text = ~paste(
      "Program:", program_short,
      "<br>Gender:", GENDER,
      "<br>Average GPA:", round(avg_gpa, 2),
      "<br>Median GPA:", round(median_gpa, 2),
      "<br>SD:", round(sd_gpa, 2),
      "<br>Sample size:", count
    )
  ) %>%
  layout(
    title = "Interactive: Average GPA by Program and Gender",
    xaxis = list(title = ""),
    yaxis = list(title = "Average GPA"),
    barmode = "group"
  )

# Create a new interactive plot showing degree class distribution by gender
interactive_degree_plot <- grads_data %>%
  group_by(GENDER, degree_class) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(GENDER) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  plot_ly(
    x = ~degree_class,
    y = ~percentage,
    color = ~GENDER,
    colors = gender_palette,
    type = "bar",
    hoverinfo = "text",
    text = ~paste(
      "Degree Class:", degree_class,
      "<br>Gender:", GENDER,
      "<br>Percentage:", round(percentage, 1), "%",
      "<br>Count:", count
    )
  ) %>%
  layout(
    title = "Degree Class Distribution by Gender",
    xaxis = list(title = ""),
    yaxis = list(title = "Percentage (%)"),
    barmode = "group"
  )

# Save the interactive plots
htmlwidgets::saveWidget(interactive_plot, "Z:\\PROJECT\\dafhe\\interactive_gpa_plot.html")
htmlwidgets::saveWidget(interactive_degree_plot, "Z:\\PROJECT\\dafhe\\interactive_degree_plot.html")