import os
import joblib

# Absoliutus kelias iki modelių katalogo
modeliu_katalogas = r"C:\Users\Dell\Desktop\Dirbtinis intelektas ir Python pagrindai\Santuokos nutraukimas\modeliai"

# Patikriname ar katalogas egzistuoja
if not os.path.isdir(modeliu_katalogas):
    print(f"❌ Nerastas katalogas: {modeliu_katalogas}")
    exit(1)

print(f"🔍 Tikriname modelių katalogą: {modeliu_katalogas}\n")

for failas in os.listdir(modeliu_katalogas):
    if failas.endswith('.joblib'):
        kelias = os.path.join(modeliu_katalogas, failas)
        print(f"📦 Tikrinamas failas: {failas}")
        try:
            objektas = joblib.load(kelias)
            print(f"✅ Įkeltas objektas: {type(objektas)}")

            if isinstance(objektas, dict):
                raktai = objektas.keys()
                print("  🔑 Žinomi raktai objekte:")
                for raktas in raktai:
                    print(f"   - {raktas}: {type(objektas[raktas])}")
            elif "sklearn" in str(type(objektas)).lower():
                print("  📊 Tai tikėtina sklearn modelis arba transformatorius.")
            else:
                print("  ℹ️ Neatpažinta struktūra.")

        except Exception as e:
            print(f"❌ Klaida įkeliant: {e}")
        print()
