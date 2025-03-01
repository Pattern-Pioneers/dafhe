# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)

# Read the data
data <- read_csv("z:/PROJECT/dafhe/real stats/degree_enrollment_gender.csv")

# Create a focused dataset for all programs from 2007-2015
enrollment_data <- data %>%
  filter(year >= 2007 & year <= 2015) %>%
  select(year,
         `CS-MAJ-M`, `CS-MAJ-F`,
         `CS-MAN-M`, `CS-MAN-F`,
         `IT-MAJ-M`, `IT-MAJ-F`) %>%
  rename(
    "CS Major (M)" = `CS-MAJ-M`,
    "CS Major (F)" = `CS-MAJ-F`,
    "CS Management (M)" = `CS-MAN-M`,
    "CS Management (F)" = `CS-MAN-F`,
    "IT Major (M)" = `IT-MAJ-M`,
    "IT Major (F)" = `IT-MAJ-F`
  ) %>%
  pivot_longer(
    cols = -year,
    names_to = "Program_Gender",
    values_to = "Enrollment"
  ) %>%
  separate(Program_Gender, into = c("Program", "Gender"), sep = " \\(") %>%
  mutate(Gender = gsub("\\)", "", Gender))

# Create enhanced plot
print(ggplot(enrollment_data, aes(x = year, y = Enrollment, color = Program, linetype = Gender)) +
  # Add smoothed line
  geom_smooth(method = "loess", se = FALSE, span = 0.75, size = 1.2) +
  # Add points
  geom_point(size = 3, alpha = 0.7) +
  # Add value labels with better positioning
  geom_text(aes(label = Enrollment),
            vjust = -0.8,
            size = 3,
            fontface = "bold") +
  # Custom colors with better opacity
  scale_color_manual(values = c(
    "CS Major" = "#4169E1",
    "CS Management" = "#FF69B4",
    "IT Major" = "#32CD32"
  )) +
  # Enhanced labels
  labs(
    title = "Computer Science and IT Enrollment Trends by Program and Gender",
    subtitle = "Period: 2007-2015",
    x = "Academic Year",
    y = "Number of New Students",
    color = "Program",
    linetype = "Gender"
  ) +
  # Improved theme
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey40"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey90"),
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
  ) +
  # Set appropriate y-axis breaks
  scale_y_continuous(
    breaks = pretty_breaks(n = 8),
    expand = expansion(mult = c(0.1, 0.2))
  )
)

# Save high-resolution plot
ggsave("program_enrollment_trends.png",
       width = 16,
       height = 9,
       dpi = 300,
       bg = "white")