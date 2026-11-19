from streamlit.testing.v1 import AppTest

from modules import llm_processor

SAMPLE = open("samples/service_agreement.txt", encoding="utf-8").read()


def test_app_renders_all_tabs_with_mocked_claude(monkeypatch):
    monkeypatch.setattr(llm_processor, "summarize_text", lambda text, model=None: "SUMMARY OK")
    monkeypatch.setattr(llm_processor, "extract_key_info",
                        lambda text, model=None: {k: (["Sarah Ahmed"] if k == "people" else [])
                                                  for k in llm_processor.KEY_INFO_FIELDS})
    monkeypatch.setattr(llm_processor, "answer_question", lambda c, q, model=None: "ANSWER OK")

    at = AppTest.from_file("../app/app_streamlit.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Upload" in i.value for i in at.info)  # empty state

    at.sidebar.text_area[0].input(SAMPLE).run()
    assert not at.exception and not at.error
    assert any("SUMMARY OK" in m.value for m in at.markdown)
    assert at.dataframe[0].value.iloc[0]["value"] == "Sarah Ahmed"

    at.text_input[0].input("Who is the project manager?").run()
    assert any("ANSWER OK" in s.value for s in at.success)
