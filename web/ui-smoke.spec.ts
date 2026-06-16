import { test, expect } from "@playwright/test"

test("analyzes the default review through the local API", async ({ page }) => {
  await page.goto("http://127.0.0.1:5173/")
  await expect(page.getByRole("heading", { name: "Turismy review analyzer" })).toBeVisible()
  await page.getByRole("button", { name: "Analyze review" }).click()
  await expect(page.getByText("TF-IDF + Linear SVM")).toBeVisible()
  await expect(page.getByText("TF-IDF + Random Forest Regressor")).toBeVisible()
  await expect(page.getByText("out of 5.0")).toBeVisible()
  await page.screenshot({ path: "../artifacts/figures/frontend-verification.png", fullPage: true })
})

test("keeps the prediction workflow usable on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto("http://127.0.0.1:5173/")
  await expect(page.getByRole("heading", { name: "Turismy review analyzer" })).toBeVisible()
  await page.getByRole("button", { name: "Analyze review" }).click()
  await expect(page.getByText("TF-IDF + Linear SVM")).toBeVisible()
  await expect(page.getByText("Visitor rating", { exact: true })).toBeVisible()
  await page.screenshot({ path: "../artifacts/figures/frontend-mobile-verification.png", fullPage: true })
})
