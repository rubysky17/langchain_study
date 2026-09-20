import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk (sinh ngày 28 tháng 6 năm 1971) là một doanh nhân, nổi tiếng với vai trò mấu chốt trong hai công ty Tesla, Inc. và SpaceX, cũng như chủ sở hữu của Twitter. Musk là người giàu nhất thế giới tính đến tháng 1 năm 2025; tính đến tháng 6 năm 2026, Forbes ước tính giá trị tài sản ròng của ông là 1,3 nghìn tỷ USD, biến ông trở thành tỷ phú nghìn tỷ đầu tiên và duy nhất trên thế giới tính theo USD.

Musk sinh ra và lớn lên trong một gia đình giàu có ở Pretoria, Nam Phi, trước khi di cư đến Canada và nhập tịch nước này. Ông chuyển tới California vào năm 1995 để theo học Đại học Stanford. Tại đây, ông cùng người em trai Kimbal đồng sáng lập công ty phần mềm Zip2, sau được mua lại bởi Compaq vào năm 1999. Cùng năm đó, Musk đồng sáng lập X.com, một ngân hàng trực tiếp về sau sáp nhập thành PayPal. Năm 2002, Musk nhập tịch Mỹ, cùng thời điểm lúc eBay mua lại PayPal. Sử dụng số tiền thu được sau thương vụ, Musk thành lập công ty dịch vụ hàng không vũ trụ SpaceX vào năm 2002. Năm 2004, Musk đầu tư sớm vào công ty sản xuất xe điện Tesla, rồi thăng tiến và trở thành chủ tịch cũng như CEO của công ty này. Năm 2018, Ủy ban Giao dịch và Chứng khoán Hoa Kỳ (SEC) đâm đơn kiện Musk với cáo buộc rằng ông đã bịa đặt về việc đảm bảo được một khoản vay nhằm thâu tóm tư nhân Tesla, buộc Musk phải từ chức chủ tịch công ty và nộp một khoản tiền phạt. Năm 2022, ông mua lại Twitter, đổi tên nền tảng thành X vào năm sau. Tháng 1 năm 2025, Musk được cắt cử làm nhân viên đặc biệt của Bộ Hiệu quả Chính phủ trong nhiệm kỳ thứ hai của Tổng thống Hoa Kỳ Donald Trump.

Musk là nhà tài trợ lớn nhất trong cuộc tranh cử Tổng thống Hoa Kỳ 2024 và được đánh giá là người ủng hộ các đảng phái, nhà hoạt động có tư tưởng cực hữu. Đầu năm 2025, ông làm cố vấn cao cấp cho Tổng thống tân cử Donald Trump và trở thành chủ nhiệm trên danh nghĩa của DOGE. Sau một cuộc tranh cãi công khai trên mạng với Trump, Musk từ bỏ chính quyền Trump và tuyên bố sẽ thành lập đảng riêng mang tên Đảng Nước Mỹ.

Những hoạt động và quan điểm chính trị mà Musk bày tỏ đã tạo dựng hình ảnh của ông như một nhân vật gây phân cực. Ông đã bị chỉ trích vì nhiều phát ngôn phi khoa học và sai sự thật, bao gồm: lan truyền thông tin sai lệch về COVID-19, bình luận cổ súy phân biệt chủng tộc, miệt thị Semit và cộng đồng chuyển giới, cũng như rao giảng các thuyết âm mưu vô căn cứ. Việc ông mua lại Twitter đã dấy lên tranh cãi vì chính sách cắt giảm nhân sự khổng lồ, sự gia tăng phát ngôn gây thù ghét và sự lan truyền thông tin sai lệch trên nền tảng này. Vai trò của ông trong chính quyền nhiệm kỳ hai của Trump, cụ thể là cơ quan DOGE, đã phải hứng chịu phản ứng dữ dội từ dư luận Hoa Kỳ.
    """

    summary_template = """
        given the information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(
        model="qwen/qwen3-8b",
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        temperature=0,
    )
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
