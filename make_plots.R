## IO Problem Set 1 Q3 (a) + (b)
## Lei Ma, Feb 2021

rm(list = ls())
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
library(tidyverse)
library(readxl)
library(AER)
library(stargazer)
library(fastDummies)
library(RColorBrewer)

otc <- read_excel("OTC_data.xlsx")
otc <- otc %>%
  mutate(
    Brand = case_when(
      product %in% c(1, 2, 3) ~ "Tylenol",
      product %in% c(4, 5, 6) ~ "Advil",
      product %in% c(7, 8, 9) ~ "Bayer",
      product %in% c(10, 11) ~ "Store brand"
    ),
    Brand = factor(Brand,
      levels = c("Tylenol", "Advil", "Bayer", "Store brand")
    )
  )
otc <- dummy_cols(otc, select_columns = "product", remove_first_dummy = TRUE)

## (a) Graph sales by brand
otc_sales_brand <- otc %>%
  group_by(Brand, week) %>%
  summarise(sales = sum(sales))
ggplot(
  otc_sales_brand,
  aes(x = week, y = sales, group = Brand, color = Brand)
) +
  geom_line() +
  geom_point(size = 1) +
  theme_minimal() +
  scale_color_brewer(palette = "Pastel2") +
  theme(legend.position = "bottom") +
  xlab("Week") +
  ylab("Sales")
ggsave("q3_sales_by_brand.png", width = 6, height = 4)


## (b) Logit model without random coefficients
otc <- otc %>%
  as.data.frame() %>%
  mutate(mkt = dplyr::group_indices(., store, week)) %>%
  group_by(store, week) %>%
  arrange(store, week, product) %>%
  mutate(mkt_size = count,
         mkt_share = sales/mkt_size,
         mkt_share_tot = sum(mkt_share),
         mkt_share_outside = 1 - mkt_share_tot,
         ln_mkt_share_diff = log(mkt_share) - log(mkt_share_outside))
write.csv(otc, "otc.csv")

m1 <- lm(ln_mkt_share_diff ~ price + promotion , data = otc)
m2 <- lm(ln_mkt_share_diff ~ price + promotion + factor(product), data = otc)
m3 <- lm(ln_mkt_share_diff ~ price + promotion + factor(product) + factor(store), data = otc)
m4 <- ivreg(ln_mkt_share_diff ~ price + promotion | . - price + cost, data = otc)
m5 <- ivreg(ln_mkt_share_diff ~ price + promotion + factor(product) | . - price + cost, data = otc)
stargazer(m1, m2, m3, m4, m5, 
          omit = c("product", "store"),
          type = "text")

## (f) Graph price over time at each store by product
mycolors <- colorRampPalette(brewer.pal(8, "Pastel1"))(11)
ggplot(otc %>% mutate(product = factor(product)),
       aes(x = week, y = price, group = product, color = product)) +
  geom_line() +
  facet_wrap(store ~ ., nrow = 2) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  xlab("Week") +
  ylab("Price") +
  scale_color_manual(values = mycolors)
ggsave("q4_price.png", width = 6, height = 6)
